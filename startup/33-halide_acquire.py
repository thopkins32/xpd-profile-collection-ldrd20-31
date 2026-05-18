"""
Acquisition plan for the halide perovskite QueueserverAgent.

This plan follows the Blop AcquisitionPlan protocol signature::

    def __call__(suggestions, actuators, sensors, md=None) -> uid

It performs the full synthesis + measurement sequence for one optimization step:

1. Set pump infusion rates (from suggestion DOF values), start pumps, wait for
   flow equilibrium. Optionally start toluene dilution.
2. Collect absorbance spectra.
3. Collect fluorescence spectra. When ``USE_GOOD_BAD`` is enabled, additional
   PL batches are taken (in the same Bluesky run) until either ``GOOD_TARGET``
   good batches or ``MAX_BAD`` bad batches have been classified.
4. Stop all pumps that were started (guaranteed even on exception).
5. Return the run UID.

Design (rewritten from first principles):

* :func:`steady_state_flow` is a wrapper plan that owns pump setup and
  teardown using ``bpp.finalize_wrapper``. Pumps stopped on success and
  on exception.
* :func:`measure_absorbance` / :func:`measure_pl` are pure measurement
  sub-plans: configure optics, then ``trigger_and_read`` N times into a
  named stream.
* :class:`PLQualityMonitor` is a ``CallbackBase`` subscribed locally via
  ``bpp.subs_decorator``. It runs synchronously in the RunEngine thread
  between event docs, so the plan can read ``monitor.good_count`` /
  ``monitor.bad_count`` immediately after each batch without any
  synchronization.
* The decision policy (continue / stop) is inline in
  :func:`_pl_with_quality_gate`.

This file is loaded into the queueserver environment via startup. All devices
(``qepro``, ``LED``, ``UV_shutter``, pump objects) and helper plans
(``start_group_infuse``) are available as globals from earlier startup files.
"""

import numpy as np
import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp
from bluesky.callbacks import CallbackBase
from ophyd import Signal


# ---------------------------------------------------------------------------
# Static configuration (physical setup — update per beamtime)
# ---------------------------------------------------------------------------

# Syringe sizes (mL) for each pump, in order matching DOF order
SYRINGE_LIST = [50, 50, 50]

# Syringe materials
SYRINGE_MATER_LIST = ["steel", "steel", "steel"]

# Target volumes (format: "value unit")
TARGET_VOL_LIST = ["30 ml", "30 ml", "30 ml"]

# Whether to auto-set target for each pump
SET_TARGET_LIST = [True, True, True]

# Rate unit
RATE_UNIT = "ul/min"

# Mixer tubing: list of length_cm for each mixer segment.
# Used to compute residence time from rates directly.
MIXER_LENGTHS_CM = [30.0]

# Inner diameter of mixer tubing (mm)
TUBING_ID_MM = 1.016

# Residence time multiplier (wait this many multiples of the residence time)
RESIDENT_T_RATIO = 1.0

# Number of absorbance and fluorescence spectra per measurement
NUM_ABS = 10
NUM_FLU = 10

# Precursor names (for metadata only)
PRECURSOR_LIST = ["CsPbOA", "TOABr", "ZnI2"]

# Post-dilution with toluene
POST_DILUTE = False
POST_DILUTE_RATIO = 1.0  # toluene rate = sum(active_rates) * ratio
POST_DILUTE_WAIT_SEC = 30  # wait time after starting toluene pump

# Default mapping: DOF name -> pump device name in queueserver namespace
DOF_TO_PUMP = {
    "infusion_rate_CsPb": "dds2_p1",
    "infusion_rate_Br": "dds2_p2",
    "infusion_rate_I2": "dds1_p1",
    "infusion_rate_Cl": "dds1_p1",
    "infusion_rate_OAm": "dds1_p2",
}

# Toluene dilution pump device name
DILUTE_PUMP_NAME = "dds1_p2"


# ---------------------------------------------------------------------------
# Good/bad fluorescence reacquisition
# ---------------------------------------------------------------------------
# When enabled, after each batch of NUM_FLU PL shots the most recent qepro
# spectrum is classified by `_classify_pl`. Additional batches are taken (in
# the same Bluesky run, into the same 'fluorescence' stream) until either
# GOOD_TARGET good batches or MAX_BAD bad batches are accumulated. One small
# bookkeeping event per batch is emitted into a single auxiliary stream
# 'fluorescence_quality' for traceability.

USE_GOOD_BAD = False
GOOD_TARGET = 3            # success once this many good batches collected
MAX_BAD = 3                # give up after this many bad batches (log + proceed)

# Classifier thresholds (legacy good_bad_data parity).
DEFAULT_THRESHOLDS = {
    "key_height": 2000,        # c1: highest peak intensity > 400 nm
    "prominence": 30,          # scipy.find_peaks prominence (legacy 'height')
    "distance": 30,            # scipy.find_peaks distance
    "integral_low": 100000,    # c2 (peak < 560 nm)
    "integral_high": 200000,   # c3 (peak >= 560 nm)
    "led_band": (340.0, 400.0),  # excluded from peak search; integrated separately
    "split_wavelength": 560.0,
    "peak_search_min_nm": 400.0,
}


# ---------------------------------------------------------------------------
# Module-level cached Signals for the 'fluorescence_quality' stream.
# Created once at import time so we don't churn descriptor UIDs across runs.
# ---------------------------------------------------------------------------

_Q_BATCH_INDEX = Signal(name="batch_index", value=0)
_Q_VERDICT = Signal(name="verdict", value="bad")
_Q_PEAK_WL = Signal(name="peak_wavelength_nm", value=float("nan"))
_Q_N_GOOD = Signal(name="n_good_total", value=0)
_Q_N_BAD = Signal(name="n_bad_total", value=0)
_Q_N_EVENTS = Signal(name="n_events_in_batch", value=0)
_Q_SIGS = [_Q_BATCH_INDEX, _Q_VERDICT, _Q_PEAK_WL, _Q_N_GOOD, _Q_N_BAD, _Q_N_EVENTS]


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


def _classify_pl(x, y, thresholds=None):
    """Classify a PL spectrum as good/bad.

    Minimal in-plan port of ``scripts/utils/_data_analysis.good_bad_data`` so
    this module has no dependency on the legacy Kafka/ZMQ pipeline. The
    classifier rejects (returns ``(False, ...)``) when any of:

    - **c1** highest peak (wavelength > ``peak_search_min_nm``, excluding the
      LED band) has intensity below ``key_height``.
    - **c2** highest peak is below ``split_wavelength`` and
      ``(PL_integral - LED_integral) < integral_low``.
    - **c3** highest peak is at/above ``split_wavelength`` and
      ``(PL_integral - LED_integral) < integral_high``.

    Parameters
    ----------
    x, y : array_like
        Wavelength (nm) and intensity arrays from the QEPro.
    thresholds : dict | None
        Threshold dict; falls back to ``DEFAULT_THRESHOLDS``.

    Returns
    -------
    (is_good, peak_wavelength_nm) : tuple[bool, float]
        ``peak_wavelength_nm`` is ``NaN`` when no peak is found.
    """
    from scipy.signal import find_peaks

    t = thresholds or DEFAULT_THRESHOLDS
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    led_lo, led_hi = t["led_band"]
    search_mask = (x > t["peak_search_min_nm"]) & ~((x >= led_lo) & (x <= led_hi))
    xs, ys = x[search_mask], y[search_mask]
    if xs.size == 0:
        return False, float("nan")

    peaks, _ = find_peaks(ys, prominence=t["prominence"], distance=t["distance"])
    if peaks.size == 0:
        return False, float("nan")

    top = peaks[int(np.argmax(ys[peaks]))]
    top_wl = float(xs[top])
    top_int = float(ys[top])

    # c1
    if top_int < t["key_height"]:
        return False, top_wl

    # Integrals over the full spectrum and the LED band.
    led_mask = (x >= led_lo) & (x <= led_hi)
    pl_int = float(np.trapz(y, x))
    led_int = float(np.trapz(y[led_mask], x[led_mask])) if led_mask.any() else 0.0
    delta = pl_int - led_int

    # c2 / c3
    if top_wl < t["split_wavelength"] and delta < t["integral_low"]:
        return False, top_wl
    if top_wl >= t["split_wavelength"] and delta < t["integral_high"]:
        return False, top_wl

    return True, top_wl


# ---------------------------------------------------------------------------
# Quality monitor (local, blocking callback)
# ---------------------------------------------------------------------------


class PLQualityMonitor(CallbackBase):
    """Subscribe via ``bpp.subs_decorator`` so it runs synchronously on the
    RunEngine thread between event docs.

    The plan reads ``.good_count`` / ``.bad_count`` after each batch and calls
    :meth:`finalize_batch` to classify the most recent spectrum and update the
    counters.

    Parameters
    ----------
    qepro : ophyd device
        The QEPro device whose ``x_axis`` / ``output`` keys carry the spectrum
        in the event document. Used to derive event-data field names so this
        module does not hardcode string keys.
    stream_name : str
        Name of the event stream to monitor (default: ``"fluorescence"``).
    thresholds : dict | None
        Classifier thresholds; defaults to ``DEFAULT_THRESHOLDS``.
    """

    def __init__(self, qepro, stream_name="fluorescence", thresholds=None):
        super().__init__()
        self.stream_name = stream_name
        self.x_field = qepro.x_axis.name
        self.y_field = qepro.output.name
        self.thresholds = thresholds or DEFAULT_THRESHOLDS

        self._target_descriptors = set()
        self._latest_spectrum = None        # (x, y) of last event in stream
        self._batch_event_count = 0

        self.good_count = 0
        self.bad_count = 0
        self.batch_index = 0
        self.batch_results = []             # list[dict], in chronological order

    def descriptor(self, doc):
        if doc.get("name") == self.stream_name:
            self._target_descriptors.add(doc["uid"])

    def event(self, doc):
        if doc["descriptor"] not in self._target_descriptors:
            return
        data = doc["data"]
        if self.x_field not in data or self.y_field not in data:
            return
        self._latest_spectrum = (
            np.asarray(data[self.x_field]),
            np.asarray(data[self.y_field]),
        )
        self._batch_event_count += 1

    def finalize_batch(self):
        """Classify the most recent spectrum in the just-finished batch.

        Returns the per-batch result dict (also appended to
        :attr:`batch_results`). Returns ``None`` if no event was observed in
        the batch (which would indicate a misconfiguration).
        """
        if self._latest_spectrum is None:
            return None
        x, y = self._latest_spectrum
        is_good, peak_wl = _classify_pl(x, y, self.thresholds)
        if is_good:
            self.good_count += 1
        else:
            self.bad_count += 1
        result = {
            "batch_index": self.batch_index,
            "verdict": "good" if is_good else "bad",
            "peak_wavelength_nm": float(peak_wl),
            "n_good_total": self.good_count,
            "n_bad_total": self.bad_count,
            "n_events_in_batch": self._batch_event_count,
        }
        self.batch_results.append(result)
        self.batch_index += 1
        self._batch_event_count = 0
        self._latest_spectrum = None
        return result


# ---------------------------------------------------------------------------
# Measurement sub-plans
# ---------------------------------------------------------------------------


def measure_absorbance(qepro, n_shots, *, stream="absorbance", settle_sec=2):
    """Configure optics for absorbance and trigger ``n_shots`` reads.

    Each shot becomes one event in the named stream.
    """
    yield from bps.mv(
        qepro.correction, "Reference",
        qepro.spectrum_type, "Absorbtion",
    )
    yield from bps.mv(LED, "Low", UV_shutter, "High")
    yield from bps.sleep(settle_sec)
    for _ in range(n_shots):
        yield from bps.trigger_and_read([qepro], name=stream)


def measure_pl(qepro, n_shots, *, stream="fluorescence", settle_sec=2):
    """Configure optics for PL and trigger ``n_shots`` reads.

    Each shot becomes one event in the named stream.
    """
    yield from bps.mv(
        qepro.correction, "Dark",
        qepro.spectrum_type, "Corrected Sample",
    )
    yield from bps.mv(LED, "High", UV_shutter, "Low")
    yield from bps.sleep(settle_sec)
    for _ in range(n_shots):
        yield from bps.trigger_and_read([qepro], name=stream)


def _emit_quality_event(result):
    """Emit one event in the ``fluorescence_quality`` stream from a result dict."""
    if result is None:
        return
    yield from bps.mv(
        _Q_BATCH_INDEX, int(result["batch_index"]),
        _Q_VERDICT, result["verdict"],
        _Q_PEAK_WL, float(result["peak_wavelength_nm"]),
        _Q_N_GOOD, int(result["n_good_total"]),
        _Q_N_BAD, int(result["n_bad_total"]),
        _Q_N_EVENTS, int(result["n_events_in_batch"]),
    )
    yield from bps.create(name="fluorescence_quality")
    for s in _Q_SIGS:
        yield from bps.read(s)
    yield from bps.save()


def _pl_with_quality_gate(qepro, monitor):
    """Run PL batches until good/bad termination.

    Always runs at least one batch. When ``monitor`` is ``None`` (i.e.,
    quality gating disabled), returns after that single batch. Otherwise
    keeps running batches until ``good_count >= GOOD_TARGET`` or
    ``bad_count >= MAX_BAD``.
    """
    # First batch always runs.
    yield from measure_pl(qepro, NUM_FLU)

    if monitor is None:
        return

    yield from _emit_quality_event(monitor.finalize_batch())

    while (monitor.good_count < GOOD_TARGET
           and monitor.bad_count < MAX_BAD):
        yield from measure_pl(qepro, NUM_FLU)
        yield from _emit_quality_event(monitor.finalize_batch())

    if monitor.good_count >= GOOD_TARGET:
        print(f"*** {monitor.good_count} good PL batches, proceeding ***")
    else:
        print(
            f"*** {monitor.bad_count} bad PL batches, "
            "proceeding anyway ***"
        )


# ---------------------------------------------------------------------------
# Steady-state flow context (setup + guaranteed teardown)
# ---------------------------------------------------------------------------


def steady_state_flow(
    plan,
    pump_list,
    rate_list,
    *,
    syringe_list,
    target_vol_list,
    set_target_list,
    syringe_mater_list,
    rate_unit=RATE_UNIT,
    resident_t_ratio=RESIDENT_T_RATIO,
    post_dilute=False,
    dilute_pump=None,
    dilute_rate_ratio=POST_DILUTE_RATIO,
    dilute_wait_sec=POST_DILUTE_WAIT_SEC,
):
    """Wrap ``plan`` with pump setup before and pump stop after.

    On entry: set rates, start the synthesis pumps, wait for the residence
    time computed from the rates, optionally start toluene dilution.

    On exit (success **or** exception): stop every pump that was started.

    This is the idiomatic ``bpp.finalize_wrapper`` pattern; the closure
    ``started_pumps`` records what needs stopping so teardown is correct
    regardless of where setup failed.
    """
    started_pumps = []

    def setup():
        # 1. Set per-pump infusion parameters.
        for pump, rate, syringe, target_vol, set_target, material in zip(
            pump_list, rate_list, syringe_list, target_vol_list,
            set_target_list, syringe_mater_list,
        ):
            if rate == 0.0:
                continue
            yield from pump.set_infuse2(
                syringe,
                set_target=set_target,
                target_vol=float(target_vol.split(" ")[0]),
                target_unit=target_vol.split(" ")[1],
                infuse_rate=rate,
                infuse_unit=rate_unit,
                syringe_material=material,
            )

        # 2. Start synthesis pumps; record which ones we started.
        yield from start_group_infuse(pump_list, rate_list)
        started_pumps.extend(
            p for p, r in zip(pump_list, rate_list) if r > 0
        )

        # 3. Wait for flow equilibrium (residence time from geometry).
        wait_sec = _compute_equilibrium_wait(rate_list, resident_t_ratio)
        print(f"\nResidence time: {wait_sec:.1f} s (ratio={resident_t_ratio})")
        yield from _sleep_with_progress(wait_sec)

        # 4. Optional toluene dilution.
        if post_dilute and dilute_pump is not None:
            toluene_rate = sum(r for r in rate_list if r > 0) * dilute_rate_ratio
            yield from dilute_pump.set_infuse2(
                50,
                set_target=True,
                target_vol=30,
                target_unit="ml",
                infuse_rate=toluene_rate,
                infuse_unit=rate_unit,
                syringe_material="steel",
            )
            yield from dilute_pump.infuse_pump2()
            started_pumps.append(dilute_pump)
            print(
                f"\nStarted toluene dilution at {toluene_rate:.1f} uL/min, "
                f"waiting {dilute_wait_sec}s"
            )
            yield from bps.sleep(dilute_wait_sec)

    def teardown():
        for p in started_pumps:
            try:
                yield from p.stop_pump2()
            except Exception as e:  # noqa: BLE001 -- best-effort cleanup
                print(f"Warning: failed to stop pump {p.name}: {e}")

    def body():
        yield from setup()
        return (yield from plan)

    return (yield from bpp.finalize_wrapper(body(), teardown()))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def halide_acquire(suggestions, actuators, sensors=None, md=None):
    """Acquire UV-Vis data for halide perovskite optimization.

    The Blop ``AcquisitionPlan`` for one optimizer suggestion:

    1. Set pump rates + start pumps (inside :func:`steady_state_flow`).
    2. Wait for flow equilibrium.
    3. Optionally start toluene dilution.
    4. In a single Bluesky run, collect absorbance and fluorescence streams.
       With ``USE_GOOD_BAD`` enabled, PL is gated by :class:`PLQualityMonitor`
       and additional batches are taken until good/bad termination.
    5. Stop all started pumps (guaranteed by ``bpp.finalize_wrapper``).
    6. Return the run UID.

    Parameters
    ----------
    suggestions : list[dict]
        Blop suggestions. Each dict maps DOF names to suggested values.
        Typically ``len(suggestions) == 1``.
    actuators : list[str]
        Pump device names (may be empty if DOFs have no actuator field).
    sensors : list[str] | None
        Sensor device names (e.g., ``["QEPro"]``).
    md : dict | None
        Extra metadata (e.g. ``blop_correlation_uid``).

    Returns
    -------
    str
        UID of the Bluesky run.
    """
    # -- Parse suggestion --
    suggestion = suggestions[0]
    dof_names = sorted(k for k in suggestion.keys() if k.startswith("infusion_rate"))
    rate_list = [float(suggestion[name]) for name in dof_names]
    pump_list = _resolve_pumps_from_dofs(dof_names)
    sample_type = _make_sample_name(rate_list)

    _md = {
        "sample_type": sample_type,
        "infuse_rates": rate_list,
        "dof_names": dof_names,
        "precursors": PRECURSOR_LIST[: len(pump_list)],
        "pumps": [p.name for p in pump_list],
        "detectors": ["qepro"],
        "use_good_bad": USE_GOOD_BAD,
    }
    _md.update(md or {})

    # -- Quality monitor (only when gating is enabled) --
    monitor = (
        PLQualityMonitor(qepro, stream_name="fluorescence")
        if USE_GOOD_BAD else None
    )
    # bpp.subs_decorator dispatches docs synchronously on the RE thread, so
    # `monitor.finalize_batch()` immediately after `bps.save()` sees the
    # just-emitted event with no synchronization required.
    subs = {"descriptor": [monitor], "event": [monitor]} if monitor else {}

    # -- Acquisition body (single Bluesky run) --
    @bpp.subs_decorator(subs)
    @bpp.stage_decorator([qepro])
    @bpp.run_decorator(md=_md)
    def acquisition():
        yield from measure_absorbance(qepro, NUM_ABS)
        yield from _pl_with_quality_gate(qepro, monitor)
        # Lights off at end of run.
        yield from bps.mv(LED, "Low", UV_shutter, "Low")

    # -- Resolve dilution pump if requested --
    dilute_pump = _resolve_pumps([DILUTE_PUMP_NAME])[0] if POST_DILUTE else None

    # -- Run the body inside the flow context (pumps guaranteed to stop) --
    uid = yield from steady_state_flow(
        acquisition(),
        pump_list=pump_list,
        rate_list=rate_list,
        syringe_list=SYRINGE_LIST[: len(pump_list)],
        target_vol_list=TARGET_VOL_LIST[: len(pump_list)],
        set_target_list=SET_TARGET_LIST[: len(pump_list)],
        syringe_mater_list=SYRINGE_MATER_LIST[: len(pump_list)],
        post_dilute=POST_DILUTE,
        dilute_pump=dilute_pump,
    )
    return uid


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _compute_equilibrium_wait(rate_list, ratio=1.0):
    """Compute equilibrium wait time (seconds) from pump rates.

    Uses mixer geometry to calculate residence time without reading hardware.

    Parameters
    ----------
    rate_list : list[float]
        Infusion rates in uL/min.
    ratio : float
        Multiplier on residence time.

    Returns
    -------
    float
        Wait time in seconds.
    """
    total_rate_ul_min = sum(r for r in rate_list if r > 0)
    if total_rate_ul_min <= 0:
        return 0.0

    # Total mixer volume in uL (pi * r^2 * length, with mm units -> mm^3 == uL).
    total_vol_ul = 0.0
    for length_cm in MIXER_LENGTHS_CM:
        length_mm = length_cm * 10.0
        radius_mm = TUBING_ID_MM / 2.0
        vol_mm3 = np.pi * radius_mm**2 * length_mm  # mm^3 == uL
        total_vol_ul += vol_mm3

    residence_time_sec = (total_vol_ul / total_rate_ul_min) * 60.0
    return residence_time_sec * ratio


def _sleep_with_progress(total_sec, steps=100):
    """Sleep for ``total_sec``, yielding periodically (mirrors ``sleep_sec_q``)."""
    if total_sec <= 0:
        return
    step_sec = total_sec / steps
    for _ in range(steps):
        yield from bps.sleep(step_sec)


def _make_sample_name(rate_list):
    """Generate a sample name from pump rates."""
    parts = [f"{r:.1f}" for r in rate_list]
    return "_".join(parts)


def _resolve_pumps(pump_names):
    """Resolve pump device objects from string names in the startup namespace."""
    pumps = []
    for name in pump_names:
        device = globals().get(name)
        if device is None:
            raise ValueError(f"Pump device '{name}' not found in queueserver namespace")
        pumps.append(device)
    return pumps


def _resolve_pumps_from_dofs(dof_names):
    """Map DOF names to pump devices via :data:`DOF_TO_PUMP`."""
    pump_names = []
    for dof in dof_names:
        pname = DOF_TO_PUMP.get(dof)
        if pname is None:
            raise ValueError(f"No pump mapping for DOF '{dof}'")
        pump_names.append(pname)
    return _resolve_pumps(pump_names)
