"""
Acquisition plan for the halide perovskite QueueserverAgent.

This plan follows the Blop AcquisitionPlan protocol signature::

    def __call__(suggestions, actuators, sensors, md=None) -> uid

It performs the full synthesis + measurement sequence for one optimization step:
1. Set pump infusion rates (from suggestion DOF values)
2. Start pumps
3. Wait for flow equilibrium (residence time, computed from rates directly)
4. Optionally start toluene dilution pump and wait
5. Collect absorbance spectra
6. Collect fluorescence spectra
7. Return the run UID

This file is loaded into the queueserver environment via startup. All devices
(qepro, LED, UV_shutter, pump objects) and helper plans (start_group_infuse,
stop_group) are available as globals from earlier startup files.
"""

import numpy as np
import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp
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

# Mixer tubing: list of (length_cm,) for each mixer segment
# Used to compute residence time from rates directly.
MIXER_LENGTHS_CM = [30.0]

# Inner diameter of mixer tubing (mm)
TUBING_ID_MM = 1.016

# Residence time multiplier (wait this many multiples of the residence time)
RESIDENT_T_RATIO = 1.0

# Number of absorbance and fluorescence spectra per measurement
NUM_ABS = 10
NUM_FLU = 10

# ---------------------------------------------------------------------------
# Good/bad fluorescence reacquisition (legacy macro_10_good_bad / macro_17 port)
# ---------------------------------------------------------------------------
# When enabled, after each batch of NUM_FLU fluorescence shots the in-memory
# qepro spectrum is classified by _classify_pl. Additional batches are taken
# (in the same Bluesky run, into the same 'fluorescence' stream) until either
# GOOD_TARGET good batches or MAX_BAD bad batches are accumulated. One small
# bookkeeping event per batch is emitted into a single auxiliary stream
# 'fluorescence_quality' for traceability.
USE_GOOD_BAD = False
GOOD_TARGET = 3            # success once this many good batches collected
MAX_BAD = 3                # give up after this many bad batches (log + proceed)
GB_KEY_HEIGHT = 2000       # c1 threshold: highest peak intensity > 400 nm
GB_PROMINENCE = 30         # scipy.find_peaks prominence (legacy 'height')
GB_DISTANCE = 30           # scipy.find_peaks distance
GB_INTEGRAL_LOW = 100000   # c2 threshold (peak < 560 nm)
GB_INTEGRAL_HIGH = 200000  # c3 threshold (peak >= 560 nm)
GB_LED_BAND = (340.0, 400.0)  # excluded from peak search; integrated separately

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
# Acquisition plan
# ---------------------------------------------------------------------------


def halide_acquire(suggestions, actuators, sensors=None, md=None):
    """Acquire UV-Vis data for halide perovskite optimization.

    This is the acquisition plan submitted by QueueserverAgent. It:
    1. Extracts pump rates from the suggestion
    2. Sets and starts pumps
    3. Waits for equilibrium (computed from rates, no hardware read)
    4. Optionally starts toluene dilution
    5. Collects absorbance + fluorescence spectra in a single run
    6. Returns the run UID

    Parameters
    ----------
    suggestions : list[dict]
        List of suggestions from the optimizer. Each dict contains DOF names
        as keys with their suggested values. Typically len == 1.
    actuators : list[str]
        Pump device names (may be empty if DOFs have no actuator field).
    sensors : list[str] | None
        Sensor device names (e.g., ["QEPro"]).
    md : dict | None
        Metadata dict (contains blop_correlation_uid for tracking).

    Yields
    ------
    Msg
        Bluesky messages.

    Returns
    -------
    str
        The UID of the Bluesky run.
    """
    suggestion = suggestions[0]

    # Extract rates from suggestion — DOF names like "infusion_rate_CsPb"
    dof_names = sorted(k for k in suggestion.keys() if k.startswith("infusion_rate"))
    rate_list = [float(suggestion[name]) for name in dof_names]

    # Resolve pump devices
    pump_list = _resolve_pumps_from_dofs(dof_names)

    sample_type = _make_sample_name(rate_list)

    # Build metadata
    _md = {
        "sample_type": sample_type,
        "infuse_rates": rate_list,
        "dof_names": dof_names,
        "precursors": PRECURSOR_LIST[: len(pump_list)],
        "pumps": [p.name for p in pump_list],
        "detectors": ["qepro"],
    }
    _md.update(md or {})

    # --- Step 1: Set pump infusion rates ---
    for pump, rate, syringe, target_vol, set_target, material in zip(
        pump_list,
        rate_list,
        SYRINGE_LIST[: len(pump_list)],
        TARGET_VOL_LIST[: len(pump_list)],
        SET_TARGET_LIST[: len(pump_list)],
        SYRINGE_MATER_LIST[: len(pump_list)],
    ):
        if rate == 0.0:
            continue
        yield from pump.set_infuse2(
            syringe,
            set_target=set_target,
            target_vol=float(target_vol.split(" ")[0]),
            target_unit=target_vol.split(" ")[1],
            infuse_rate=rate,
            infuse_unit=RATE_UNIT,
            syringe_material=material,
        )

    # --- Step 2: Start pumps ---
    yield from start_group_infuse(pump_list, rate_list)

    # --- Step 3: Wait for equilibrium ---
    # Compute residence time directly from rate_list (no hardware read needed)
    wait_sec = _compute_equilibrium_wait(rate_list, RESIDENT_T_RATIO)
    print(f"\nResidence time: {wait_sec:.1f} s (ratio={RESIDENT_T_RATIO})")
    yield from _sleep_with_progress(wait_sec)

    # --- Step 4: Optional toluene post-dilution ---
    dilute_pump = None
    if POST_DILUTE:
        dilute_pump = _resolve_pumps([DILUTE_PUMP_NAME])[0]
        toluene_rate = sum(rate_list) * POST_DILUTE_RATIO
        yield from dilute_pump.set_infuse2(
            50,
            set_target=True,
            target_vol=30,
            target_unit="ml",
            infuse_rate=toluene_rate,
            infuse_unit=RATE_UNIT,
            syringe_material="steel",
        )
        yield from dilute_pump.infuse_pump2()
        print(
            f"\nStarted toluene dilution at {toluene_rate:.1f} uL/min, waiting {POST_DILUTE_WAIT_SEC}s"
        )
        yield from bps.sleep(POST_DILUTE_WAIT_SEC)

    # --- Step 5: Collect absorbance + fluorescence in a single run ---
    uid = yield from _acquire_uvvis(_md)

    # --- Step 6: Stop dilution pump if active ---
    if dilute_pump is not None:
        yield from dilute_pump.stop_pump2()

    return uid


# ---------------------------------------------------------------------------
# UV-Vis collection
# ---------------------------------------------------------------------------


def _acquire_uvvis(md):
    """Collect absorbance and fluorescence spectra in a single Bluesky run.

    Produces two streams: 'absorbance' and 'fluorescence'.
    Mirrors startup/32-bundle-plan.py xray_uvvis_plan2 (without X-ray).
    """

    @bpp.stage_decorator([qepro])
    @bpp.run_decorator(md=md)
    def _inner():
        # --- Absorbance ---
        yield from bps.mv(
            qepro.correction,
            "Reference",
            qepro.spectrum_type,
            "Absorbtion",
        )
        yield from bps.mv(LED, "Low", UV_shutter, "High")
        yield from bps.sleep(2)

        for _ in range(NUM_ABS):
            yield from bps.trigger(qepro, wait=True)
            yield from bps.create(name="absorbance")
            yield from bps.read(qepro)
            yield from bps.save()

        # --- Fluorescence (with optional good/bad reacquisition) ---
        yield from bps.mv(
            qepro.correction,
            "Dark",
            qepro.spectrum_type,
            "Corrected Sample",
        )
        yield from bps.mv(LED, "High", UV_shutter, "Low")
        yield from bps.sleep(2)

        # Auxiliary Signals for the per-batch bookkeeping stream.
        # Created lazily so they only exist when the plan runs.
        q_batch_index = Signal(name="batch_index", value=0)
        q_verdict = Signal(name="verdict", value="bad")
        q_peak_wl = Signal(name="peak_wavelength_nm", value=float("nan"))
        q_n_good = Signal(name="n_good_total", value=0)
        q_n_bad = Signal(name="n_bad_total", value=0)
        q_sigs = [q_batch_index, q_verdict, q_peak_wl, q_n_good, q_n_bad]

        good_count = 0
        bad_count = 0
        batch_index = 0
        # Bound the loop so a misconfigured classifier can't run forever.
        max_batches = (GOOD_TARGET + MAX_BAD) if USE_GOOD_BAD else 1

        while batch_index < max_batches:
            # One batch of NUM_FLU shots into the 'fluorescence' stream.
            # All retry batches share the same stream name (identical
            # descriptor), so no stream proliferation occurs.
            for _ in range(NUM_FLU):
                yield from bps.trigger(qepro, wait=True)
                yield from bps.create(name="fluorescence")
                yield from bps.read(qepro)
                yield from bps.save()

            if not USE_GOOD_BAD:
                break

            # Snapshot the just-acquired spectrum from the device cache and
            # classify it (legacy behavior: classifier sees one spectrum per
            # batch, not an average across the NUM_FLU shots).
            x = np.asarray(qepro.x_axis.get())
            y = np.asarray(qepro.output.get())
            is_good, peak_wl = _classify_pl(x, y)
            if is_good:
                good_count += 1
            else:
                bad_count += 1

            # Emit one bookkeeping event in 'fluorescence_quality'.
            yield from bps.mv(
                q_batch_index, batch_index,
                q_verdict, "good" if is_good else "bad",
                q_peak_wl, float(peak_wl),
                q_n_good, good_count,
                q_n_bad, bad_count,
            )
            yield from bps.create(name="fluorescence_quality")
            for s in q_sigs:
                yield from bps.read(s)
            yield from bps.save()

            batch_index += 1
            if good_count >= GOOD_TARGET:
                print(
                    f"*** Got {good_count} good fluorescence batches, proceeding ***"
                )
                break
            if bad_count >= MAX_BAD:
                print(
                    f"*** {bad_count} bad fluorescence batches, giving up "
                    "(proceeding anyway) ***"
                )
                break

        # --- Lights off ---
        yield from bps.mv(LED, "Low", UV_shutter, "Low")

    return (yield from _inner())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _classify_pl(x, y):
    """Classify a PL spectrum as good/bad.

    Minimal in-plan port of ``scripts/utils/_data_analysis.good_bad_data`` so
    this module has no dependency on the legacy Kafka/ZMQ pipeline. The
    classifier rejects (returns ``(False, ...)``) when any of:

    - **c1** highest peak (wavelength > 400 nm, excluding the LED band)
      has intensity below ``GB_KEY_HEIGHT``.
    - **c2** highest peak is < 560 nm and
      ``(PL_integral - LED_integral) < GB_INTEGRAL_LOW``.
    - **c3** highest peak is >= 560 nm and
      ``(PL_integral - LED_integral) < GB_INTEGRAL_HIGH``.

    Parameters
    ----------
    x, y : array_like
        Wavelength (nm) and intensity arrays from the QEPro.

    Returns
    -------
    (is_good, peak_wavelength_nm) : tuple[bool, float]
        ``peak_wavelength_nm`` is ``NaN`` when no peak is found.
    """
    from scipy.signal import find_peaks

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Restrict peak search to >400 nm and exclude the LED band.
    led_lo, led_hi = GB_LED_BAND
    search_mask = (x > 400.0) & ~((x >= led_lo) & (x <= led_hi))
    xs, ys = x[search_mask], y[search_mask]
    if xs.size == 0:
        return False, float("nan")

    peaks, _ = find_peaks(ys, prominence=GB_PROMINENCE, distance=GB_DISTANCE)
    if peaks.size == 0:
        return False, float("nan")

    top = peaks[int(np.argmax(ys[peaks]))]
    top_wl = float(xs[top])
    top_int = float(ys[top])

    # c1
    if top_int < GB_KEY_HEIGHT:
        return False, top_wl

    # Integrals over the full spectrum and the LED band.
    led_mask = (x >= led_lo) & (x <= led_hi)
    pl_int = float(np.trapz(y, x))
    led_int = float(np.trapz(y[led_mask], x[led_mask])) if led_mask.any() else 0.0
    delta = pl_int - led_int

    # c2 / c3
    if top_wl < 560.0 and delta < GB_INTEGRAL_LOW:
        return False, top_wl
    if top_wl >= 560.0 and delta < GB_INTEGRAL_HIGH:
        return False, top_wl

    return True, top_wl


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

    # Total mixer volume in uL (pi * r^2 * length, converted from mm/cm to uL)
    total_vol_ul = 0.0
    for length_cm in MIXER_LENGTHS_CM:
        length_mm = length_cm * 10.0
        radius_mm = TUBING_ID_MM / 2.0
        vol_mm3 = np.pi * radius_mm**2 * length_mm  # mm^3 == uL
        total_vol_ul += vol_mm3

    residence_time_sec = (total_vol_ul / total_rate_ul_min) * 60.0
    return residence_time_sec * ratio


def _sleep_with_progress(total_sec, steps=100):
    """Sleep for total_sec, yielding periodically (mirrors sleep_sec_q)."""
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
    """Resolve pump device objects from their string names in the startup namespace."""
    pumps = []
    for name in pump_names:
        device = globals().get(name)
        if device is None:
            raise ValueError(f"Pump device '{name}' not found in queueserver namespace")
        pumps.append(device)
    return pumps


def _resolve_pumps_from_dofs(dof_names):
    """Map DOF names to pump devices via DOF_TO_PUMP mapping."""
    pump_names = []
    for dof in dof_names:
        pname = DOF_TO_PUMP.get(dof)
        if pname is None:
            raise ValueError(f"No pump mapping for DOF '{dof}'")
        pump_names.append(pname)
    return _resolve_pumps(pump_names)
