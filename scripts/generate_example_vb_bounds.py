"""Reproduce the bundled example virtual-battery (VB) flexibility bands.

flex-depot is the *commercialisation* layer: it consumes depot-level VB
flexibility bands as an input. The bands themselves are produced by the
aggregation method of Park et al. (see the paper for the formal derivation).
That method is published as a methods paper without an accompanying code
release, so this stand-alone script exists solely to make the two bundled
example inputs fully reproducible:

    data/example/flexibility/vb_bounds_example.csv       (bidirectional, V2G)
    data/example/flexibility/vb_bounds_example_uni.csv   (unidirectional)

Both bands describe the same electrified truck fleet (fleet_test_id = 3,
572 kWh battery, 350 kW charge power, eta = 0.98). They differ in a single
parameter: the maximum discharge power (0 kW for the unidirectional case,
350 kW for the bidirectional case). Everything else is identical.

The script is self-contained (only pandas + numpy) and deliberately lives
outside the ``flex_dep_opt`` package: it is a reproducibility helper, not part
of the commercialisation tool.

Inputs (fleet-3 slice of the public raw data, bundled under
``data/example/flexibility/source/``; see the README there for provenance):

    tracks_with_energy_fleet3.csv                     raw driving tracks
    fleet_fleet3.csv                                  vehicle -> fleet mapping
    activities_constant_charging_350-350_fleet3.csv   SoC reconstruction

Usage::

    python scripts/generate_example_vb_bounds.py                 # regenerate both
    python scripts/generate_example_vb_bounds.py --case bidi     # only bidi
    python scripts/generate_example_vb_bounds.py --case uni      # only uni
    python scripts/generate_example_vb_bounds.py --check         # verify vs bundled

Method reference:
    Park et al., "Scalable EV Flexibility Aggregation with Guaranteed
    Disaggregation via Fast Projection onto Reachable Sets: Application to Real
    Truck Fleet Data" (methods paper, under review).
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "data" / "example" / "flexibility" / "source"
FLEXIBILITY_DIR = PROJECT_ROOT / "data" / "example" / "flexibility"

TRACKS_FILE = SOURCE_DIR / "tracks_with_energy_fleet3.csv"
FLEET_FILE = SOURCE_DIR / "fleet_fleet3.csv"
PAPER_SOC_FILE = SOURCE_DIR / "activities_constant_charging_350-350_fleet3.csv"

FLEXBAND_TIMEZONE = "Europe/Berlin"

# --------------------------------------------------------------------------- #
# Column names (raw inputs and intermediate frames)
# --------------------------------------------------------------------------- #

TRACK_ID = "track_id"
VEHICLE_ID = "vehicle_id"
FLEET_ID = "fleet_test_id"
START_TIME = "start_time"
STOP_TIME = "stop_time"
HOME_BASE = "home_base"
ENERGY_KWH_CLEANED = "energy_consumption_kwh_cleaned"

ARRIVAL_TIME = "arrival_time"
DEPARTURE_TIME = "departure_time"
DURATION_H = "duration_h"
NEXT_TRACK_ID = "next_track_id"
NEXT_TRIP_ENERGY_KWH = "next_trip_energy_kwh"
REQUIRED_ENERGY_KWH = "required_energy_kwh"
ARRIVAL_ENERGY_KWH = "arrival_energy_kwh"
ARRIVAL_SOC = "arrival_soc"
MIN_DEPARTURE_ENERGY_KWH = "min_departure_energy_kwh"
MIN_ENERGY_KWH = "min_energy_kwh"
MAX_ENERGY_KWH = "max_energy_kwh"

TIMESTAMP = "timestamp"
ACTIVE_SESSIONS = "active_sessions"
P_LOWER_KW = "p_lower_kw"
P_UPPER_KW = "p_upper_kw"
E_LOWER_KWH = "e_lower_kwh"
E_UPPER_KWH = "e_upper_kwh"
REF_DRIVING_ENERGY_KWH = "ref_driving_energy_kwh"

# Energy bands narrower than this are collapsed points whose lower/upper values
# only differ by float noise from separate arithmetic paths.
ENERGY_BAND_SNAP_TOLERANCE_KWH = 1e-6


# --------------------------------------------------------------------------- #
# Paper configuration (frozen; corresponds to config/default.toml of the
# aggregation repo, fleet 3). Only ``max_discharge_power_kw`` varies between the
# unidirectional and bidirectional example.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PaperConfig:
    max_discharge_power_kw: float  # 0.0 = unidirectional, 350.0 = bidirectional

    # Vehicle
    battery_capacity_kwh: float = 572.0
    max_charge_power_kw: float = 350.0
    min_soc: float = 0.0
    max_soc: float = 1.0
    target_departure_soc: float = 1.0
    charging_efficiency: float = 0.98
    discharging_efficiency: float = 0.98

    # Sessions
    min_parking_duration_minutes: int = 15
    departure_energy_reserve_kwh: float = 0.0

    # Time / grid
    resolution_minutes: int = 15
    flexband_start_date: str = "2026-01-01"

    # Flexbands
    carry_forward_energy_after_departure: bool = True

    # Fleet-3 track window (inclusive end); mirrors [fleet_time_filters.3].
    fleet3_end: str = "2022-03-21T23:45:00+01:00"


CASES = {"uni": 0.0, "bidi": 350.0}
CASE_OUTPUT = {
    "uni": FLEXIBILITY_DIR / "vb_bounds_example_uni.csv",
    "bidi": FLEXIBILITY_DIR / "vb_bounds_example.csv",
}
# Collapsed energy bands (lower == upper up to float noise) are snapped to an
# exact single point only for the unidirectional file: it was exported through
# the aggregation repo's main pipeline, which applies the snap. The
# bidirectional file was exported through that repo's sensitivity pipeline,
# which does not snap, so its collapsed bands keep their raw last-ULP spread.
# Reproducing both files byte-for-byte therefore requires this per-case toggle.
CASE_SNAP_COLLAPSED = {"uni": True, "bidi": False}


# --------------------------------------------------------------------------- #
# Session construction
# --------------------------------------------------------------------------- #

def _normalize_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    result = normalized.map(mapping)
    unknown = sorted(series[result.isna()].dropna().astype(str).unique())
    if unknown:
        raise ValueError(f"Unknown boolean values in {series.name}: {unknown}")
    return result.astype(bool)


def _attach_fleet_ids(tracks: pd.DataFrame, fleet: pd.DataFrame) -> pd.DataFrame:
    vehicle_fleet = fleet[[VEHICLE_ID, FLEET_ID]].drop_duplicates()
    merged = tracks.merge(
        vehicle_fleet, on=VEHICLE_ID, how="left", validate="many_to_one"
    )
    missing = sorted(merged.loc[merged[FLEET_ID].isna(), VEHICLE_ID].unique().tolist())
    if missing:
        raise ValueError(f"Tracks without fleet assignment for vehicle_id: {missing}")
    return merged


def _apply_fleet3_time_filter(tracks: pd.DataFrame, cfg: PaperConfig) -> pd.DataFrame:
    filtered = tracks.copy()
    filtered[START_TIME] = pd.to_datetime(filtered[START_TIME], format="mixed", utc=True)
    filtered[STOP_TIME] = pd.to_datetime(filtered[STOP_TIME], format="mixed", utc=True)
    end = pd.Timestamp(cfg.fleet3_end).tz_convert("UTC")
    fleet_mask = filtered[FLEET_ID].astype(int) == 3
    keep = ~fleet_mask | (filtered[STOP_TIME] <= end)
    return filtered[keep].reset_index(drop=True)


def _set_paper_soc_energy_states(tracks: pd.DataFrame, cfg: PaperConfig) -> pd.DataFrame:
    """Set depot arrival energy from the activity-level SoC reconstruction."""
    paper_soc = pd.read_csv(PAPER_SOC_FILE)
    paper_home = paper_soc[paper_soc["occupation"] == "home base"].copy()
    paper_home[START_TIME] = pd.to_datetime(
        paper_home[START_TIME], format="mixed", utc=True
    )
    paper_home["paper_arrival_soc"] = pd.to_numeric(
        paper_home["soc_start"], errors="coerce"
    )
    paper_home = paper_home[[VEHICLE_ID, START_TIME, "paper_arrival_soc"]].dropna()
    paper_home = paper_home.drop_duplicates(
        subset=[VEHICLE_ID, START_TIME], keep="first"
    )

    prepared = tracks.merge(
        paper_home,
        how="left",
        left_on=[VEHICLE_ID, STOP_TIME],
        right_on=[VEHICLE_ID, START_TIME],
        suffixes=("", "_paper"),
    )
    prepared = prepared.drop(columns=[f"{START_TIME}_paper"])

    arrival_soc = prepared["paper_arrival_soc"].clip(lower=cfg.min_soc, upper=cfg.max_soc)
    prepared[ARRIVAL_SOC] = arrival_soc
    prepared[ARRIVAL_ENERGY_KWH] = arrival_soc * cfg.battery_capacity_kwh
    prepared = prepared.drop(columns=["paper_arrival_soc"])
    return prepared


def build_depot_sessions(
    tracks_with_fleets: pd.DataFrame, cfg: PaperConfig
) -> pd.DataFrame:
    """Build depot sessions from home-base arrivals and each vehicle's next trip."""
    tracks = tracks_with_fleets.copy()
    tracks[HOME_BASE] = _normalize_bool(tracks[HOME_BASE])
    tracks[START_TIME] = pd.to_datetime(tracks[START_TIME], format="mixed", utc=True)
    tracks[STOP_TIME] = pd.to_datetime(tracks[STOP_TIME], format="mixed", utc=True)
    tracks = tracks.sort_values([VEHICLE_ID, START_TIME, STOP_TIME]).reset_index(drop=True)

    tracks = _set_paper_soc_energy_states(tracks, cfg)

    grouped = tracks.groupby(VEHICLE_ID, sort=False)
    tracks[DEPARTURE_TIME] = grouped[START_TIME].shift(-1)
    tracks[NEXT_TRACK_ID] = grouped[TRACK_ID].shift(-1)
    tracks[NEXT_TRIP_ENERGY_KWH] = grouped[ENERGY_KWH_CLEANED].shift(-1)

    sessions = tracks[tracks[HOME_BASE] & tracks[DEPARTURE_TIME].notna()].copy()
    sessions[ARRIVAL_TIME] = sessions[STOP_TIME]
    sessions[DURATION_H] = (
        sessions[DEPARTURE_TIME] - sessions[ARRIVAL_TIME]
    ).dt.total_seconds() / 3600.0

    min_duration_h = cfg.min_parking_duration_minutes / 60.0
    sessions = sessions[sessions[DURATION_H] >= min_duration_h].copy()
    sessions = sessions[sessions[DURATION_H] > 0].copy()
    if sessions[ARRIVAL_ENERGY_KWH].isna().any():
        missing = sessions.loc[
            sessions[ARRIVAL_ENERGY_KWH].isna(), [VEHICLE_ID, TRACK_ID, ARRIVAL_TIME]
        ].head(5)
        raise ValueError(
            "Missing arrival energy for depot sessions. First unmatched rows:\n"
            f"{missing.to_string(index=False)}"
        )
    sessions[NEXT_TRACK_ID] = sessions[NEXT_TRACK_ID].astype("int64")

    max_energy_kwh = cfg.battery_capacity_kwh * cfg.max_soc
    sessions[MIN_ENERGY_KWH] = cfg.battery_capacity_kwh * cfg.min_soc
    sessions[MAX_ENERGY_KWH] = max_energy_kwh

    # departure energy strategy = "full_charge"
    target_departure_energy_kwh = cfg.battery_capacity_kwh * cfg.target_departure_soc
    sessions[MIN_DEPARTURE_ENERGY_KWH] = target_departure_energy_kwh
    sessions[REQUIRED_ENERGY_KWH] = (
        sessions[MIN_DEPARTURE_ENERGY_KWH] - sessions[ARRIVAL_ENERGY_KWH]
    ).clip(lower=0.0)

    sessions[MIN_DEPARTURE_ENERGY_KWH] = (
        sessions[MIN_DEPARTURE_ENERGY_KWH] + cfg.departure_energy_reserve_kwh
    )
    sessions[MIN_DEPARTURE_ENERGY_KWH] = np.minimum(
        sessions[MIN_DEPARTURE_ENERGY_KWH], sessions[MAX_ENERGY_KWH]
    )

    return sessions[
        [
            VEHICLE_ID,
            FLEET_ID,
            ARRIVAL_TIME,
            DEPARTURE_TIME,
            REQUIRED_ENERGY_KWH,
            ARRIVAL_ENERGY_KWH,
            MIN_DEPARTURE_ENERGY_KWH,
            MIN_ENERGY_KWH,
            MAX_ENERGY_KWH,
        ]
    ].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Per-vehicle robust envelopes and fleet aggregation (Park et al., Sec. 2.2)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class VehicleEnvelopeInputs:
    arrival_index: int
    departure_index: int
    arrival_energy_kwh: float
    min_departure_energy_kwh: float
    min_energy_kwh: float
    max_energy_kwh: float
    max_charge_power_kw: float
    max_discharge_power_kw: float
    charging_efficiency: float
    discharging_efficiency: float
    timestep_hours: float


@dataclass(frozen=True)
class VehicleEnvelope:
    start_index: int
    end_index: int
    lower_kwh: np.ndarray
    upper_kwh: np.ndarray
    max_charge_power_kw: float
    max_discharge_power_kw: float


@dataclass(frozen=True)
class VirtualBatteryBounds:
    lower_energy_kwh: np.ndarray
    upper_energy_kwh: np.ndarray
    lower_power_kw: np.ndarray
    upper_power_kw: np.ndarray
    active_sessions: np.ndarray


def build_vehicle_envelope(inputs: VehicleEnvelopeInputs) -> VehicleEnvelope:
    """Build robust per-session cumulative energy bounds relative to arrival."""
    steps = inputs.departure_index - inputs.arrival_index
    lower = np.zeros(steps, dtype=float)
    upper = np.zeros(steps, dtype=float)
    available_discharge_kwh = inputs.arrival_energy_kwh - inputs.min_energy_kwh
    charge_headroom_kwh = inputs.max_energy_kwh - inputs.arrival_energy_kwh

    charge_step_kwh = (
        inputs.max_charge_power_kw * inputs.charging_efficiency * inputs.timestep_hours
    )
    discharge_step_kwh = (
        inputs.max_discharge_power_kw / inputs.discharging_efficiency * inputs.timestep_hours
    )
    max_reachable_charge_kwh = min(charge_headroom_kwh, charge_step_kwh * steps)
    required_charge_kwh = min(
        inputs.min_departure_energy_kwh - inputs.arrival_energy_kwh,
        max_reachable_charge_kwh,
    )

    previous_lower = 0.0
    previous_upper = 0.0

    for offset in range(steps):
        elapsed_h = (offset + 1) * inputs.timestep_hours
        remaining_h = (steps - 1 - offset) * inputs.timestep_hours

        raw_upper = min(
            charge_headroom_kwh,
            inputs.max_charge_power_kw * inputs.charging_efficiency * elapsed_h,
        )
        lower_forward = max(
            -available_discharge_kwh,
            -inputs.max_discharge_power_kw / inputs.discharging_efficiency * elapsed_h,
        )
        lower_backward = max(
            -available_discharge_kwh,
            required_charge_kwh
            - inputs.max_charge_power_kw * inputs.charging_efficiency * remaining_h,
        )
        raw_lower = max(lower_forward, lower_backward)

        robust_upper = min(raw_upper, previous_lower + charge_step_kwh)
        robust_lower = max(raw_lower, previous_upper - discharge_step_kwh)

        if robust_lower > robust_upper:
            robust_lower = robust_upper

        lower[offset] = robust_lower
        upper[offset] = robust_upper
        previous_lower = robust_lower
        previous_upper = robust_upper

    return VehicleEnvelope(
        start_index=inputs.arrival_index,
        end_index=inputs.departure_index,
        lower_kwh=lower,
        upper_kwh=upper,
        max_charge_power_kw=inputs.max_charge_power_kw,
        max_discharge_power_kw=inputs.max_discharge_power_kw,
    )


def aggregate_envelopes(
    envelopes: list[VehicleEnvelope],
    horizon_steps: int,
    carry_forward_energy_after_departure: bool,
) -> VirtualBatteryBounds:
    """Sum per-vehicle robust envelopes into fleet-level virtual battery bounds."""
    lower_energy = np.zeros(horizon_steps, dtype=float)
    upper_energy = np.zeros(horizon_steps, dtype=float)
    lower_power = np.zeros(horizon_steps, dtype=float)
    upper_power = np.zeros(horizon_steps, dtype=float)
    active_sessions = np.zeros(horizon_steps, dtype=np.int64)

    for envelope in envelopes:
        if not 0 <= envelope.start_index < envelope.end_index <= horizon_steps:
            raise ValueError("Envelope indices must lie inside the horizon.")

        index_slice = slice(envelope.start_index, envelope.end_index)
        lower_power[index_slice] -= envelope.max_discharge_power_kw
        upper_power[index_slice] += envelope.max_charge_power_kw
        active_sessions[index_slice] += 1

        state_start = envelope.start_index + 1
        state_end = min(envelope.end_index + 1, horizon_steps)
        active_state_steps = state_end - state_start
        if active_state_steps > 0:
            state_slice = slice(state_start, state_end)
            lower_energy[state_slice] += envelope.lower_kwh[:active_state_steps]
            upper_energy[state_slice] += envelope.upper_kwh[:active_state_steps]

        if carry_forward_energy_after_departure and envelope.end_index + 1 < horizon_steps:
            carry_slice = slice(envelope.end_index + 1, horizon_steps)
            lower_energy[carry_slice] += envelope.lower_kwh[-1]
            upper_energy[carry_slice] += envelope.upper_kwh[-1]

    return VirtualBatteryBounds(
        lower_energy_kwh=lower_energy,
        upper_energy_kwh=upper_energy,
        lower_power_kw=lower_power,
        upper_power_kw=upper_power,
        active_sessions=active_sessions,
    )


# --------------------------------------------------------------------------- #
# Time grid helpers
# --------------------------------------------------------------------------- #

def build_time_grid(
    start: pd.Timestamp, end: pd.Timestamp, resolution_minutes: int
) -> pd.DatetimeIndex:
    start_ts = pd.Timestamp(start).tz_convert("UTC").floor(f"{resolution_minutes}min")
    end_ts = pd.Timestamp(end).tz_convert("UTC").ceil(f"{resolution_minutes}min")
    if end_ts <= start_ts:
        raise ValueError("end must be after start.")
    return pd.date_range(
        start=start_ts, end=end_ts, freq=f"{resolution_minutes}min", inclusive="both"
    )


def timestamp_to_index(
    timestamp: pd.Timestamp, grid_start: pd.Timestamp, resolution_minutes: int, mode: str
) -> int:
    timestamp = pd.Timestamp(timestamp).tz_convert("UTC")
    grid_start = pd.Timestamp(grid_start).tz_convert("UTC")
    frequency = f"{resolution_minutes}min"
    aligned = timestamp.floor(frequency) if mode == "floor" else timestamp.ceil(frequency)
    delta_minutes = (aligned - grid_start).total_seconds() / 60.0
    return int(delta_minutes // resolution_minutes)


# --------------------------------------------------------------------------- #
# Flexband construction
# --------------------------------------------------------------------------- #

def build_depot_flexbands(sessions: pd.DataFrame, cfg: PaperConfig) -> pd.DataFrame:
    """Build power and energy flexbands for the fleet (cumulative net energy)."""
    if sessions.empty:
        raise ValueError("At least one depot session is required.")

    resolution = cfg.resolution_minutes
    # grid_alignment = "conservative": arrival -> ceil, departure -> floor.
    arrival_alignment, departure_alignment = "ceil", "floor"
    timestep_hours = resolution / 60.0

    prepared = sessions.copy()
    prepared[ARRIVAL_TIME] = pd.to_datetime(prepared[ARRIVAL_TIME], utc=True)
    prepared[DEPARTURE_TIME] = pd.to_datetime(prepared[DEPARTURE_TIME], utc=True)

    flexbands = []
    for fleet_id, fleet_sessions in prepared.groupby(FLEET_ID, sort=True):
        fleet_start = fleet_sessions[ARRIVAL_TIME].min()
        grid = build_time_grid(fleet_start, fleet_sessions[DEPARTURE_TIME].max(), resolution)
        grid_start = grid[0]
        envelopes: list[VehicleEnvelope] = []
        driving_energy = np.zeros(len(grid), dtype=float)

        for row in fleet_sessions.itertuples(index=False):
            arrival_time = getattr(row, ARRIVAL_TIME)
            departure_time = getattr(row, DEPARTURE_TIME)
            arrival_index = timestamp_to_index(
                arrival_time, grid_start, resolution, arrival_alignment
            )
            departure_index = timestamp_to_index(
                departure_time, grid_start, resolution, departure_alignment
            )
            if departure_index <= arrival_index:
                continue

            n_steps = departure_index - arrival_index
            driving_energy[arrival_index:departure_index] += (
                getattr(row, REQUIRED_ENERGY_KWH) / cfg.charging_efficiency / n_steps
            )

            envelopes.append(
                build_vehicle_envelope(
                    VehicleEnvelopeInputs(
                        arrival_index=arrival_index,
                        departure_index=departure_index,
                        arrival_energy_kwh=getattr(row, ARRIVAL_ENERGY_KWH),
                        min_departure_energy_kwh=getattr(row, MIN_DEPARTURE_ENERGY_KWH),
                        min_energy_kwh=getattr(row, MIN_ENERGY_KWH),
                        max_energy_kwh=getattr(row, MAX_ENERGY_KWH),
                        max_charge_power_kw=cfg.max_charge_power_kw,
                        max_discharge_power_kw=cfg.max_discharge_power_kw,
                        charging_efficiency=cfg.charging_efficiency,
                        discharging_efficiency=cfg.discharging_efficiency,
                        timestep_hours=timestep_hours,
                    )
                )
            )

        bounds = aggregate_envelopes(
            envelopes,
            horizon_steps=len(grid),
            carry_forward_energy_after_departure=cfg.carry_forward_energy_after_departure,
        )
        flexbands.append(
            pd.DataFrame(
                {
                    FLEET_ID: fleet_id,
                    TIMESTAMP: grid,
                    ACTIVE_SESSIONS: bounds.active_sessions,
                    P_LOWER_KW: bounds.lower_power_kw,
                    P_UPPER_KW: bounds.upper_power_kw,
                    E_LOWER_KWH: bounds.lower_energy_kwh,
                    E_UPPER_KWH: bounds.upper_energy_kwh,
                    REF_DRIVING_ENERGY_KWH: driving_energy,
                }
            )
        )

    result = pd.concat(flexbands, ignore_index=True)
    return _shift_flexband_start_date(result, cfg)


def _shift_flexband_start_date(flexbands: pd.DataFrame, cfg: PaperConfig) -> pd.DataFrame:
    """Move each fleet's time axis to the configured anchor, keeping weekday."""
    shifted = flexbands.copy()
    new_start = pd.Timestamp(cfg.flexband_start_date)
    if new_start.tzinfo is None:
        new_start = new_start.tz_localize(FLEXBAND_TIMEZONE)
    new_start = new_start.tz_convert("UTC")

    shifted[TIMESTAMP] = pd.to_datetime(shifted[TIMESTAMP], utc=True)
    for _, index in shifted.groupby(FLEET_ID).groups.items():
        old_start = shifted.loc[index, TIMESTAMP].min()
        old_start_local = old_start.tz_convert(FLEXBAND_TIMEZONE)
        old_weekday = old_start_local.weekday()
        new_weekday = new_start.tz_convert(FLEXBAND_TIMEZONE).weekday()
        days_to_add = (old_weekday - new_weekday) % 7
        old_date = old_start_local.date()
        new_date = new_start.tz_convert(FLEXBAND_TIMEZONE).date()
        day_delta = pd.Timedelta(days=(new_date - old_date).days + days_to_add)
        shifted.loc[index, TIMESTAMP] = shifted.loc[index, TIMESTAMP] + day_delta

    return shifted


# --------------------------------------------------------------------------- #
# Market-optimizer export format (flex-depot input schema)
# --------------------------------------------------------------------------- #

def to_market_flexband_format(
    flexbands: pd.DataFrame, snap_collapsed_bands: bool
) -> pd.DataFrame:
    """Convert internal flexband columns to the flex-depot input format."""
    market = (
        flexbands[
            [TIMESTAMP, P_LOWER_KW, P_UPPER_KW, E_LOWER_KWH, E_UPPER_KWH, REF_DRIVING_ENERGY_KWH]
        ]
        .rename(
            columns={
                TIMESTAMP: "time",
                P_LOWER_KW: "Power_lower_kW",
                P_UPPER_KW: "Power_upper_kW",
                E_LOWER_KWH: "Capacity_lower_kWh",
                E_UPPER_KWH: "Capacity_upper_kWh",
                REF_DRIVING_ENERGY_KWH: "Ref_driving_energy_kWh",
            }
        )
        .reset_index(drop=True)
    )
    market["time"] = pd.to_datetime(market["time"], utc=True).dt.tz_convert(FLEXBAND_TIMEZONE)

    if snap_collapsed_bands:
        # Collapsed energy bands: write identical lower/upper so float noise
        # cannot flip their ordering when the CSV is read back.
        width = market["Capacity_upper_kWh"] - market["Capacity_lower_kWh"]
        collapsed = width.abs() < ENERGY_BAND_SNAP_TOLERANCE_KWH
        market.loc[collapsed, "Capacity_lower_kWh"] = market.loc[
            collapsed, "Capacity_upper_kWh"
        ]
    return market


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def generate_case(case: str) -> pd.DataFrame:
    """Generate the market-format VB bands for one case ('uni' or 'bidi')."""
    cfg = PaperConfig(max_discharge_power_kw=CASES[case])
    tracks = pd.read_csv(TRACKS_FILE)
    fleet = pd.read_csv(FLEET_FILE)
    tracks_with_fleets = _attach_fleet_ids(tracks, fleet)
    tracks_with_fleets = _apply_fleet3_time_filter(tracks_with_fleets, cfg)
    sessions = build_depot_sessions(tracks_with_fleets, cfg)
    flexbands = build_depot_flexbands(sessions, cfg)
    return to_market_flexband_format(
        flexbands, snap_collapsed_bands=CASE_SNAP_COLLAPSED[case]
    )


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=["uni", "bidi", "both"],
        default="both",
        help="Which example band(s) to regenerate (default: both).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare against the bundled example files instead of overwriting them.",
    )
    args = parser.parse_args()

    cases = ["uni", "bidi"] if args.case == "both" else [args.case]
    exit_code = 0
    for case in cases:
        target = CASE_OUTPUT[case]
        generated = generate_case(case)
        if args.check:
            if not target.exists():
                print(f"[{case}] MISSING bundled file: {target}")
                exit_code = 1
                continue
            # Compare the exact CSV serialisation, normalising line endings so
            # the check is platform-independent (the bundled files use CRLF).
            generated_text = generated.to_csv(index=False, lineterminator="\n")
            expected_text = target.read_text().replace("\r\n", "\n")
            if generated_text == expected_text:
                print(f"[{case}] OK  reproduces {target.name} exactly ({len(generated)} rows)")
            else:
                print(f"[{case}] MISMATCH vs {target.name}")
                exit_code = 1
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            generated.to_csv(target, index=False)
            print(f"[{case}] wrote {len(generated)} rows -> {target}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
