#!/usr/bin/env python3
"""
Convert the ses01 final EEGLAB datasets (P01-P08) to compressed NPZ files.

Expected final EEGLAB filenames:
    Control/P##_10_remove_bad_components_2_c.set
    TMS_on_Lip_Target/P##_10_remove_bad_components_2_tl.set
    TMS_on_Tongue_Target/P##_10_remove_bad_components_2_tt.set

Each output NPZ stores:
    X                    : (trials, channels, timepoints)
    y_condition          : integer condition label per trial
    condition_per_trial  : string condition label per trial
    subject_per_trial    : subject ID per trial
    event_code           : EEGLAB/MNE event code per trial
    event_label          : event description per trial
    channel_names
    channel_types
    channel_positions_xyz : electrode coordinates in metres, shape (channels, 3)
    channel_loc           : complete MNE 12-value channel location vectors
    channel_position_valid
    channel_coordinate_frames
    dig_*                 : digitisation/fiducial metadata when present
    sfreq
    times
    unit
    source_set

A combined NPZ is also created when channel/time structures are compatible.
Missing final files are skipped without stopping the batch. The original .set/.fdt files are never modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import mne
except ImportError as exc:
    raise SystemExit(
        "MNE-Python is required.\n"
        "Install it in the active environment with:\n"
        "  python -m pip install --upgrade mne numpy\n"
    ) from exc


DEFAULT_ROOT = Path(
    "/Users/moonrozina/BCSC-Neural-Decoding/data/EEG_Data_Processing_ses01"
)
DEFAULT_OUTPUT = Path(
    "/Users/moonrozina/BCSC-Neural-Decoding/data/EEG_Data_Processing_ses01_npz"
)


@dataclass(frozen=True)
class CategorySpec:
    key: str
    condition_id: int
    suffix: str
    folder_candidates: tuple[str, ...]


CATEGORIES = (
    CategorySpec(
        key="Control",
        condition_id=0,
        suffix="_c",
        folder_candidates=("Control", "control"),
    ),
    CategorySpec(
        key="TMS_on_Lip_Target",
        condition_id=1,
        suffix="_tl",
        folder_candidates=(
            "TMS_on_Lip_Target",
            "TMS_on_Lip_Targuet",  # also accept the ses02 typo
            "tms_true_lip",
        ),
    ),
    CategorySpec(
        key="TMS_on_Tongue_Target",
        condition_id=2,
        suffix="_tt",
        folder_candidates=(
            "TMS_on_Tongue_Target",
            "tms_true_tongue",
        ),
    ),
)


def parse_subjects(text: str) -> list[int]:
    """Parse values such as '1-16', '1,3,5', or '1-4,8,10-12'."""
    numbers: set[int] = set()

    for block in text.split(","):
        block = block.strip()
        if not block:
            continue

        if "-" in block:
            start_text, end_text = block.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start > end:
                raise ValueError(f"Invalid subject range: {block}")
            numbers.update(range(start, end + 1))
        else:
            numbers.add(int(block))

    if not numbers:
        raise ValueError("No subjects were selected.")

    invalid = sorted(number for number in numbers if number < 0 or number > 999)
    if invalid:
        raise ValueError(f"Invalid subject numbers: {invalid}")

    return sorted(numbers)


def find_category_folder(root: Path, candidates: Iterable[str]) -> Path | None:
    for folder_name in candidates:
        folder = root / folder_name
        if folder.is_dir():
            return folder
    return None


def get_epochs_data(epochs: "mne.BaseEpochs") -> np.ndarray:
    """Support both recent and older MNE versions."""
    try:
        return epochs.get_data(copy=True)
    except TypeError:
        return epochs.get_data()


def event_labels_from_epochs(
    epochs: "mne.BaseEpochs",
) -> tuple[np.ndarray, np.ndarray, str]:
    n_epochs = len(epochs)

    if getattr(epochs, "events", None) is None or len(epochs.events) != n_epochs:
        event_codes = np.full(n_epochs, -1, dtype=np.int32)
    else:
        event_codes = epochs.events[:, 2].astype(np.int32, copy=False)

    event_id = {
        str(label): int(code)
        for label, code in getattr(epochs, "event_id", {}).items()
    }

    labels_by_code: dict[int, str] = {}
    for label, code in event_id.items():
        labels_by_code.setdefault(code, label)

    event_labels = np.asarray(
        [labels_by_code.get(int(code), "unknown") for code in event_codes],
        dtype=str,
    )

    event_id_json = json.dumps(
        event_id,
        ensure_ascii=False,
        sort_keys=True,
    )
    return event_codes, event_labels, event_id_json


def extract_spatial_metadata(
    epochs: "mne.BaseEpochs",
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    str,
]:
    """Extract channel coordinates and digitisation metadata from MNE Info.

    MNE stores distances in metres.  The first three entries of each channel's
    12-value ``loc`` vector are the channel XYZ coordinates.  A position is
    treated as valid only when all three values are finite and the coordinate
    is not the zero vector.
    """
    n_channels = len(epochs.ch_names)
    channel_loc = np.full((n_channels, 12), np.nan, dtype=np.float64)
    channel_coordinate_frames = np.full(n_channels, -1, dtype=np.int32)

    for channel_index, channel_info in enumerate(epochs.info["chs"]):
        loc = np.asarray(channel_info.get("loc", []), dtype=np.float64).ravel()
        n_values = min(loc.size, 12)
        if n_values:
            channel_loc[channel_index, :n_values] = loc[:n_values]
        channel_coordinate_frames[channel_index] = int(
            channel_info.get("coord_frame", -1)
        )

    channel_positions_xyz = channel_loc[:, :3].copy()
    channel_position_valid = (
        np.all(np.isfinite(channel_positions_xyz), axis=1)
        & (np.linalg.norm(channel_positions_xyz, axis=1) > 0.0)
    )

    dig_points = epochs.info.get("dig") or []
    dig_r: list[np.ndarray] = []
    dig_kind: list[int] = []
    dig_ident: list[int] = []
    dig_coordinate_frames: list[int] = []
    dig_json_records: list[dict[str, object]] = []

    for point in dig_points:
        r = np.asarray(point["r"], dtype=np.float64).reshape(3)
        kind = int(point["kind"])
        ident = int(point["ident"])
        coord_frame = int(point["coord_frame"])

        dig_r.append(r)
        dig_kind.append(kind)
        dig_ident.append(ident)
        dig_coordinate_frames.append(coord_frame)
        dig_json_records.append(
            {
                "r_m": [float(value) for value in r],
                "kind": kind,
                "ident": ident,
                "coord_frame": coord_frame,
            }
        )

    if dig_r:
        dig_r_array = np.vstack(dig_r).astype(np.float64, copy=False)
    else:
        dig_r_array = np.empty((0, 3), dtype=np.float64)

    dig_kind_array = np.asarray(dig_kind, dtype=np.int32)
    dig_ident_array = np.asarray(dig_ident, dtype=np.int32)
    dig_coordinate_frames_array = np.asarray(
        dig_coordinate_frames,
        dtype=np.int32,
    )
    dig_points_json = json.dumps(
        dig_json_records,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return (
        channel_positions_xyz,
        channel_loc,
        channel_position_valid,
        channel_coordinate_frames,
        dig_r_array,
        dig_kind_array,
        dig_ident_array,
        dig_coordinate_frames_array,
        dig_points_json,
    )


def save_subject_condition_npz(
    output_path: Path,
    *,
    x: np.ndarray,
    subject: str,
    category: CategorySpec,
    event_codes: np.ndarray,
    event_labels: np.ndarray,
    event_id_json: str,
    channel_names: list[str],
    channel_types: list[str],
    channel_positions_xyz: np.ndarray,
    channel_loc: np.ndarray,
    channel_position_valid: np.ndarray,
    channel_coordinate_frames: np.ndarray,
    dig_r: np.ndarray,
    dig_kind: np.ndarray,
    dig_ident: np.ndarray,
    dig_coordinate_frames: np.ndarray,
    dig_points_json: str,
    sfreq: float,
    times: np.ndarray,
    unit: str,
    source_set: Path,
) -> None:
    n_trials = x.shape[0]

    np.savez_compressed(
        output_path,
        X=x,
        y_condition=np.full(n_trials, category.condition_id, dtype=np.int8),
        condition_per_trial=np.full(n_trials, category.key, dtype=str),
        subject_per_trial=np.full(n_trials, subject, dtype=str),
        epoch_in_file=np.arange(n_trials, dtype=np.int32),
        event_code=event_codes,
        event_label=event_labels,
        event_id_json=np.asarray(event_id_json),
        channel_names=np.asarray(channel_names, dtype=str),
        channel_types=np.asarray(channel_types, dtype=str),
        channel_positions_xyz=np.asarray(
            channel_positions_xyz,
            dtype=np.float64,
        ),
        channel_positions_unit=np.asarray("m"),
        channel_loc=np.asarray(channel_loc, dtype=np.float64),
        channel_position_valid=np.asarray(
            channel_position_valid,
            dtype=bool,
        ),
        channel_coordinate_frames=np.asarray(
            channel_coordinate_frames,
            dtype=np.int32,
        ),
        dig_r=np.asarray(dig_r, dtype=np.float64),
        dig_kind=np.asarray(dig_kind, dtype=np.int32),
        dig_ident=np.asarray(dig_ident, dtype=np.int32),
        dig_coordinate_frames=np.asarray(
            dig_coordinate_frames,
            dtype=np.int32,
        ),
        dig_points_json=np.asarray(dig_points_json),
        sfreq=np.asarray(sfreq, dtype=np.float64),
        times=np.asarray(times, dtype=np.float64),
        tmin=np.asarray(times[0], dtype=np.float64),
        tmax=np.asarray(times[-1], dtype=np.float64),
        unit=np.asarray(unit),
        source_set=np.asarray(str(source_set)),
        subject=np.asarray(subject),
        condition=np.asarray(category.key),
        condition_id=np.asarray(category.condition_id, dtype=np.int8),
        session=np.asarray("ses01"),
    )


def compatible_with_reference(
    channel_names: list[str],
    sfreq: float,
    times: np.ndarray,
    reference_channels: list[str],
    reference_sfreq: float,
    reference_times: np.ndarray,
) -> tuple[bool, str]:
    if channel_names != reference_channels:
        return False, "channel order/names differ"
    if not np.isclose(sfreq, reference_sfreq, rtol=0.0, atol=1e-9):
        return False, "sampling frequency differs"
    if times.shape != reference_times.shape:
        return False, "number of timepoints differs"
    if not np.allclose(times, reference_times, rtol=0.0, atol=1e-9):
        return False, "epoch time axis differs"
    return True, ""


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert final EEGLAB .set/.fdt datasets into NPZ files. "
            "Missing subjects are skipped."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"EEG_Data_Processing_ses01 folder (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output folder (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--subjects",
        default="1-8",
        help="Participants to process, for example: 1-8 or 1,3,5-8",
    )
    parser.add_argument(
        "--unit",
        choices=("uV", "V"),
        default="uV",
        help=(
            "Unit stored in X. MNE loads EEG data in volts; default converts "
            "it back to microvolts (uV)."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
        help="Data type used for X (default: float32).",
    )
    parser.add_argument(
        "--no-combined",
        action="store_true",
        help="Create only per-subject/condition files, not the combined NPZ.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip an NPZ when the per-file output already exists.",
    )
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()

    try:
        subject_numbers = parse_subjects(args.subjects)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    root = args.root.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    per_file_output = output_root / "per_subject_condition"
    per_file_output.mkdir(parents=True, exist_ok=True)

    if not root.is_dir():
        print(f"[ERROR] Root folder does not exist: {root}", file=sys.stderr)
        return 2

    dtype = np.float32 if args.dtype == "float32" else np.float64
    unit_scale = 1e6 if args.unit == "uV" else 1.0

    summary_rows: list[dict[str, object]] = []

    combined_x: list[np.ndarray] = []
    combined_y: list[np.ndarray] = []
    combined_subjects: list[np.ndarray] = []
    combined_conditions: list[np.ndarray] = []
    combined_event_codes: list[np.ndarray] = []
    combined_event_labels: list[np.ndarray] = []
    combined_epoch_indices: list[np.ndarray] = []
    combined_source_sets: list[np.ndarray] = []
    combined_dataset_indices: list[np.ndarray] = []
    combined_dataset_subjects: list[str] = []
    combined_dataset_conditions: list[str] = []
    combined_dataset_source_sets: list[str] = []
    combined_dataset_channel_positions_xyz: list[np.ndarray] = []
    combined_dataset_channel_loc: list[np.ndarray] = []
    combined_dataset_channel_position_valid: list[np.ndarray] = []
    combined_dataset_channel_coordinate_frames: list[np.ndarray] = []
    combined_dataset_dig_points_json: list[str] = []

    reference_channels: list[str] | None = None
    reference_channel_types: list[str] | None = None
    reference_sfreq: float | None = None
    reference_times: np.ndarray | None = None

    processed = 0
    skipped = 0
    failed = 0
    combined_trials = 0

    print(f"Root:   {root}")
    print(f"Output: {output_root}")
    print(f"Unit:   {args.unit}")
    print(f"Dtype:  {args.dtype}\n")

    for category in CATEGORIES:
        category_folder = find_category_folder(root, category.folder_candidates)

        if category_folder is None:
            print(f"[SKIP] Category folder missing: {category.key}")
            for number in subject_numbers:
                subject = f"P{number:02d}"
                summary_rows.append(
                    {
                        "subject": subject,
                        "condition": category.key,
                        "status": "SKIPPED",
                        "message": "Category folder missing",
                        "input_set": "",
                        "output_npz": "",
                        "n_trials": "",
                        "n_channels": "",
                        "n_timepoints": "",
                        "sfreq": "",
                        "n_nan": "",
                        "included_in_combined": False,
                    }
                )
                skipped += 1
            continue

        print(f"=== {category.key} ===")
        print(category_folder)

        for number in subject_numbers:
            subject = f"P{number:02d}"
            final_name = (
                f"{subject}_10_remove_bad_components_2{category.suffix}.set"
            )
            set_path = category_folder / final_name
            output_path = per_file_output / f"{set_path.stem}.npz"

            if not set_path.is_file():
                print(f"[SKIP] {subject}: final .set not found")
                summary_rows.append(
                    {
                        "subject": subject,
                        "condition": category.key,
                        "status": "SKIPPED",
                        "message": "Final .set missing",
                        "input_set": str(set_path),
                        "output_npz": str(output_path),
                        "n_trials": "",
                        "n_channels": "",
                        "n_timepoints": "",
                        "sfreq": "",
                        "n_nan": "",
                        "included_in_combined": False,
                    }
                )
                skipped += 1
                continue

            if output_path.exists() and args.skip_existing:
                print(f"[SKIP] {subject}: NPZ already exists")
                summary_rows.append(
                    {
                        "subject": subject,
                        "condition": category.key,
                        "status": "SKIPPED",
                        "message": "NPZ already exists; --skip-existing used",
                        "input_set": str(set_path),
                        "output_npz": str(output_path),
                        "n_trials": "",
                        "n_channels": "",
                        "n_timepoints": "",
                        "sfreq": "",
                        "n_nan": "",
                        "included_in_combined": False,
                    }
                )
                skipped += 1
                continue

            try:
                epochs = mne.read_epochs_eeglab(
                    str(set_path),
                    verbose="ERROR",
                )

                current_channels = list(epochs.ch_names)

                # Standardize channel order for combined output when the same
                # channel set exists in a different order.
                if (
                    reference_channels is not None
                    and current_channels != reference_channels
                    and set(current_channels) == set(reference_channels)
                ):
                    epochs.reorder_channels(reference_channels)
                    current_channels = list(epochs.ch_names)

                x_volts = get_epochs_data(epochs)
                x = np.asarray(x_volts * unit_scale, dtype=dtype)

                if x.ndim != 3:
                    raise ValueError(
                        f"Expected a 3-D epoched array, got shape {x.shape}"
                    )

                n_trials, n_channels, n_timepoints = x.shape
                sfreq = float(epochs.info["sfreq"])
                times = np.asarray(epochs.times, dtype=np.float64)
                channel_types = list(epochs.get_channel_types())
                (
                    channel_positions_xyz,
                    channel_loc,
                    channel_position_valid,
                    channel_coordinate_frames,
                    dig_r,
                    dig_kind,
                    dig_ident,
                    dig_coordinate_frames,
                    dig_points_json,
                ) = extract_spatial_metadata(epochs)
                event_codes, event_labels, event_id_json = (
                    event_labels_from_epochs(epochs)
                )
                n_nan = int(np.isnan(x).sum())

                save_subject_condition_npz(
                    output_path,
                    x=x,
                    subject=subject,
                    category=category,
                    event_codes=event_codes,
                    event_labels=event_labels,
                    event_id_json=event_id_json,
                    channel_names=current_channels,
                    channel_types=channel_types,
                    channel_positions_xyz=channel_positions_xyz,
                    channel_loc=channel_loc,
                    channel_position_valid=channel_position_valid,
                    channel_coordinate_frames=channel_coordinate_frames,
                    dig_r=dig_r,
                    dig_kind=dig_kind,
                    dig_ident=dig_ident,
                    dig_coordinate_frames=dig_coordinate_frames,
                    dig_points_json=dig_points_json,
                    sfreq=sfreq,
                    times=times,
                    unit=args.unit,
                    source_set=set_path,
                )

                included_in_combined = False
                combined_message = ""

                if not args.no_combined:
                    if reference_channels is None:
                        reference_channels = current_channels
                        reference_channel_types = channel_types
                        reference_sfreq = sfreq
                        reference_times = times.copy()
                        is_compatible = True
                    else:
                        assert reference_sfreq is not None
                        assert reference_times is not None
                        is_compatible, combined_message = compatible_with_reference(
                            current_channels,
                            sfreq,
                            times,
                            reference_channels,
                            reference_sfreq,
                            reference_times,
                        )

                    if is_compatible:
                        dataset_index = len(combined_dataset_subjects)
                        combined_x.append(x)
                        combined_y.append(
                            np.full(
                                n_trials,
                                category.condition_id,
                                dtype=np.int8,
                            )
                        )
                        combined_subjects.append(
                            np.full(n_trials, subject, dtype=str)
                        )
                        combined_conditions.append(
                            np.full(n_trials, category.key, dtype=str)
                        )
                        combined_event_codes.append(event_codes)
                        combined_event_labels.append(event_labels)
                        combined_epoch_indices.append(
                            np.arange(n_trials, dtype=np.int32)
                        )
                        combined_source_sets.append(
                            np.full(n_trials, str(set_path), dtype=str)
                        )
                        combined_dataset_indices.append(
                            np.full(n_trials, dataset_index, dtype=np.int32)
                        )
                        combined_dataset_subjects.append(subject)
                        combined_dataset_conditions.append(category.key)
                        combined_dataset_source_sets.append(str(set_path))
                        combined_dataset_channel_positions_xyz.append(
                            channel_positions_xyz
                        )
                        combined_dataset_channel_loc.append(channel_loc)
                        combined_dataset_channel_position_valid.append(
                            channel_position_valid
                        )
                        combined_dataset_channel_coordinate_frames.append(
                            channel_coordinate_frames
                        )
                        combined_dataset_dig_points_json.append(
                            dig_points_json
                        )
                        combined_trials += n_trials
                        included_in_combined = True
                    else:
                        print(
                            f"       [WARN] Excluded from combined NPZ: "
                            f"{combined_message}"
                        )

                message = "OK"
                if n_nan:
                    message = f"OK, but X contains {n_nan} NaN values"
                if combined_message:
                    message += f"; combined exclusion: {combined_message}"

                valid_position_count = int(channel_position_valid.sum())
                print(
                    f"[DONE] {subject}: X={x.shape}, "
                    f"sfreq={sfreq:g} Hz, NaN={n_nan}, "
                    f"channel positions={valid_position_count}/{n_channels}"
                )

                summary_rows.append(
                    {
                        "subject": subject,
                        "condition": category.key,
                        "status": "PROCESSED",
                        "message": message,
                        "input_set": str(set_path),
                        "output_npz": str(output_path),
                        "n_trials": n_trials,
                        "n_channels": n_channels,
                        "n_timepoints": n_timepoints,
                        "sfreq": sfreq,
                        "n_nan": n_nan,
                        "included_in_combined": included_in_combined,
                    }
                )
                processed += 1

            except Exception as exc:
                print(f"[FAIL] {subject}: {type(exc).__name__}: {exc}")
                summary_rows.append(
                    {
                        "subject": subject,
                        "condition": category.key,
                        "status": "FAILED",
                        "message": f"{type(exc).__name__}: {exc}",
                        "input_set": str(set_path),
                        "output_npz": str(output_path),
                        "n_trials": "",
                        "n_channels": "",
                        "n_timepoints": "",
                        "sfreq": "",
                        "n_nan": "",
                        "included_in_combined": False,
                    }
                )
                failed += 1

        print()

    combined_path = output_root / "EEG_Data_Processing_ses01_all.npz"

    if not args.no_combined and combined_x:
        assert reference_channels is not None
        assert reference_channel_types is not None
        assert reference_sfreq is not None
        assert reference_times is not None

        print("Creating combined NPZ...")

        np.savez_compressed(
            combined_path,
            X=np.concatenate(combined_x, axis=0),
            y_condition=np.concatenate(combined_y, axis=0),
            subject_per_trial=np.concatenate(combined_subjects, axis=0),
            condition_per_trial=np.concatenate(combined_conditions, axis=0),
            event_code=np.concatenate(combined_event_codes, axis=0),
            event_label=np.concatenate(combined_event_labels, axis=0),
            epoch_in_source_file=np.concatenate(
                combined_epoch_indices,
                axis=0,
            ),
            source_set_per_trial=np.concatenate(
                combined_source_sets,
                axis=0,
            ),
            dataset_index_per_trial=np.concatenate(
                combined_dataset_indices,
                axis=0,
            ),
            dataset_subjects=np.asarray(
                combined_dataset_subjects,
                dtype=str,
            ),
            dataset_conditions=np.asarray(
                combined_dataset_conditions,
                dtype=str,
            ),
            dataset_source_sets=np.asarray(
                combined_dataset_source_sets,
                dtype=str,
            ),
            dataset_channel_positions_xyz=np.stack(
                combined_dataset_channel_positions_xyz,
                axis=0,
            ),
            dataset_channel_loc=np.stack(
                combined_dataset_channel_loc,
                axis=0,
            ),
            dataset_channel_position_valid=np.stack(
                combined_dataset_channel_position_valid,
                axis=0,
            ),
            dataset_channel_coordinate_frames=np.stack(
                combined_dataset_channel_coordinate_frames,
                axis=0,
            ),
            dataset_dig_points_json=np.asarray(
                combined_dataset_dig_points_json,
                dtype=str,
            ),
            channel_positions_unit=np.asarray("m"),
            channel_names=np.asarray(reference_channels, dtype=str),
            channel_types=np.asarray(reference_channel_types, dtype=str),
            sfreq=np.asarray(reference_sfreq, dtype=np.float64),
            times=np.asarray(reference_times, dtype=np.float64),
            tmin=np.asarray(reference_times[0], dtype=np.float64),
            tmax=np.asarray(reference_times[-1], dtype=np.float64),
            unit=np.asarray(args.unit),
            condition_names=np.asarray(
                [category.key for category in CATEGORIES],
                dtype=str,
            ),
            condition_ids=np.asarray(
                [category.condition_id for category in CATEGORIES],
                dtype=np.int8,
            ),
            session=np.asarray("ses01"),
        )
        print(
            f"[DONE] Combined: {combined_path}\n"
            f"       trials={combined_trials}, "
            f"channels={len(reference_channels)}, "
            f"timepoints={len(reference_times)}"
        )
    elif not args.no_combined:
        print("[WARN] No compatible datasets were available for a combined NPZ.")

    summary_path = output_root / "conversion_summary.csv"
    fieldnames = [
        "subject",
        "condition",
        "status",
        "message",
        "input_set",
        "output_npz",
        "n_trials",
        "n_channels",
        "n_timepoints",
        "sfreq",
        "n_nan",
        "included_in_combined",
    ]

    with summary_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n=== SUMMARY ===")
    print(f"Processed: {processed}")
    print(f"Skipped:   {skipped}")
    print(f"Failed:    {failed}")
    print(f"CSV log:   {summary_path}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
