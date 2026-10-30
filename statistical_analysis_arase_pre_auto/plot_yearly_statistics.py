#!/usr/bin/env python3
"""Combine monthly completed results and plot annual kappa/X0 and AE statistics.

Reads local manifests and CSVs only; never runs the notebook or downloads data.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

import batch_arase as batch
import statistics_phases
import statistics_selection as selection
from statistics_period import add_period_arguments, resolve_period


PHASES = statistics_phases.PHASES
SOURCE_PHASES = statistics_phases.SOURCE_PHASES


def prepare_year(year, ranges_dir, state_root):
    return prepare_period(pd.Timestamp(year=year, month=1, day=1),
                          pd.Timestamp(year=year + 1, month=1, day=1), ranges_dir, state_root)


def prepare_period(start_time, end_time, ranges_dir, state_root):
    """Validate and pool monthly raw ranges across an inclusive month period."""
    frames, merged_ranges, sources = [], [], []
    next_id = 1
    for start in pd.date_range(start_time, end_time, freq="MS", inclusive="left"):
        end = start + pd.offsets.MonthBegin(1)
        path = ranges_dir / (
            f"Arase_valid_time_ranges_{start:%Y%m%d}_000000_to_"
            f"{end:%Y%m%d}_000000.json"
        )
        payload = json.loads(path.read_text())
        manifest_hash = batch.file_sha256(path)
        from plot_occurrence_probability import manifest_time_window
        if manifest_time_window(payload, payload.get("ranges", [])) != (start, end):
            raise ValueError(f"Manifest month mismatch: {path}")
        source = {
            "month": f"{start:%Y-%m}", "manifest": str(path.resolve()),
            "manifest_sha256": manifest_hash,
        }
        sources.append(source)
        if payload.get("analysis_status") == "no_data":
            if payload.get("ranges") != []:
                raise ValueError(f"no_data manifest has ranges: {path}")
            source.update(status="no_data", reason=payload.get("no_data_reason"))
            continue

        _, ranges = batch.load_ranges(path)
        dataset = batch.dataset_key(path, manifest_hash)
        aggregate = state_root / dataset / "aggregate"
        master_path = aggregate / "arase_phase_master.csv"
        status_path = aggregate / "range_status.csv"
        states = pd.read_csv(status_path)
        expected = {int(item["range_id"]): item for item in ranges}
        if states["range_id"].duplicated().any() or set(states["range_id"]) != set(expected):
            raise ValueError(f"Status IDs do not match manifest: {status_path}")
        if not states["status"].isin(["complete", "excluded"]).all():
            raise ValueError(f"Unfinished ranges: {status_path}")
        versions = states["analysis_version"]
        missing_excluded = states["status"].eq("excluded") & versions.isna()
        if not (versions.eq(batch.ANALYSIS_VERSION) | missing_excluded).all():
            raise ValueError(f"Status analysis version mismatch: {status_path}")
        master = pd.read_csv(master_path)
        if not master["analysis_version"].eq(batch.ANALYSIS_VERSION).all():
            raise ValueError(f"Master analysis version mismatch: {master_path}")
        complete = set(states.loc[states["status"].eq("complete"), "range_id"])
        actual_pairs = list(zip(master["range_id"], master["phase_mode"]))
        expected_pairs = {(rid, phase) for rid in complete for phase in SOURCE_PHASES}
        if len(actual_pairs) != len(set(actual_pairs)) or set(actual_pairs) != expected_pairs:
            raise ValueError(f"Master must contain four phases per complete range: {master_path}")
        if not master["status"].isin(["ok", "insufficient_data"]).all():
            raise ValueError(f"Unexpected phase status: {master_path}")
        id_map = {}
        for item in ranges:
            rid = int(item["range_id"])
            item_start, item_end = pd.Timestamp(item["start_time"]), pd.Timestamp(item["end_time"])
            if not (start <= item_start < item_end < end):
                raise ValueError(f"Range outside month: {path}, range={rid}")
            rows = master.loc[master["range_id"].eq(rid)]
            if not (pd.to_datetime(rows["range_start"]).eq(item_start).all()
                    and pd.to_datetime(rows["range_end"]).eq(item_end).all()):
                raise ValueError(f"Master time mismatch: {master_path}, range={rid}")
            id_map[rid] = next_id
            merged_ranges.append({
                **item, "range_id": next_id, "source_range_id": rid,
                "source_dataset": dataset,
            })
            next_id += 1
        master = master.copy()
        master["source_dataset"] = dataset
        master["source_range_id"] = master["range_id"]
        master["range_uid"] = master["range_id"].map(lambda rid: f"{dataset}:{rid}")
        master["range_id"] = master["source_range_id"].map(id_map)
        frames.append(master)
        source.update(
            status="processed", dataset=dataset,
            master=str(master_path.resolve()), master_sha256=batch.file_sha256(master_path),
            range_status=str(status_path.resolve()),
            range_status_sha256=batch.file_sha256(status_path),
            excluded_without_analysis_version=int(missing_excluded.sum()),
            range_status_counts={str(k): int(v) for k, v in states["status"].value_counts().items()},
            phase_ok_counts={phase: int((master["phase_mode"].eq(phase)
                                        & master["status"].eq("ok")).sum()) for phase in SOURCE_PHASES},
        )
    if not frames:
        raise ValueError(f"No completed monthly results for {start_time} -- {end_time}")
    master = statistics_phases.classify_phases(pd.concat(frames, ignore_index=True))
    manifest = {
        "description": "Period plotting input; range IDs remapped, source IDs retained.",
        "time_range_input": [f"{start_time:%Y%m%d}/00:00:00", f"{end_time:%Y%m%d}/00:00:00"],
        "analysis_version": batch.ANALYSIS_VERSION, "ranges": merged_ranges,
    }
    provenance = {
        "year": start_time.year if (start_time.month == 1 and end_time == start_time + pd.DateOffset(years=1)) else None,
        "start_month": f"{start_time:%Y%m}",
        "end_month": f"{end_time - pd.Timedelta(1, unit='ns'):%Y%m}",
        "analysis_version": batch.ANALYSIS_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(), "sources": sources,
        "weighting": "one completed range per sample within each phase",
        "missing_months_are_zero_detections": False,
        "master_rows": len(master),
        "direction_classification": {
            "latitude_column": "range_MLAT_deg_mean",
            "positive_mlat": {"north_traveling": "toward_ionosphere", "south_traveling": "toward_magnetic_equator"},
            "negative_mlat": {"north_traveling": "toward_magnetic_equator", "south_traveling": "toward_ionosphere"},
            "unclassified_row_counts": {
                str(k): int(v) for k, v in master.loc[
                    master["phase_mode"].eq("unclassified"), "direction_classification_reason"
                ].value_counts().items()
            },
            "phase_ok_counts": {phase: int((master["phase_mode"].eq(phase)
                                            & master["status"].eq("ok")).sum()) for phase in PHASES},
        },
    }
    return master, manifest, provenance


def build_parser():
    import plot_kappa_x0_statistics as spatial

    parser = argparse.ArgumentParser(description=__doc__)
    add_period_arguments(parser)
    selection.add_selection_argument(parser)
    spatial.add_bin_count_argument(parser)
    parser.add_argument("--ranges-dir", type=Path, default=batch.DEFAULT_RANGES.parent)
    parser.add_argument("--state-root", type=Path, default=batch.DEFAULT_STATE_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--phase", action="append", choices=PHASES)
    parser.add_argument("--prepare-only", action="store_true", help="Validate and merge without rendering.")
    parser.add_argument("--refresh-inputs", action="store_true",
                        help="Rebuild shared period inputs from monthly results; default: reuse verified prepared inputs.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import plot_kappa_x0_statistics as spatial
    import plot_ae_parameter_statistics as ae

    spatial.PLOT_CONFIG["min_bin_count"] = args.min_bin_count
    period = resolve_period(args)
    period_root = args.output_dir or batch.DEFAULT_OUTPUT_ROOT / "statistics" / period.key
    output = period_root / selection.selection_key(args.min_occupied_blocks)
    inputs = period_root / "inputs"
    manifest_path = inputs / "annual_ranges.json"
    master_path = inputs / "arase_phase_master.csv"
    reused = not args.refresh_inputs and all((inputs / name).is_file() for name in [
        "annual_ranges.json", "arase_phase_master.csv", "provenance.json"])
    if reused:
        master, provenance = selection.load_prepared_inputs(period_root, period)
        print(f"Reused prepared inputs: {inputs}", flush=True)
    else:
        master, manifest, provenance = prepare_period(period.start, period.end, args.ranges_dir, args.state_root)
    selected = selection.select_events(master, args.min_occupied_blocks)
    phases = list(dict.fromkeys(args.phase or PHASES))
    for phase in phases:
        spatial._select_phase(selected, phase)
        ae.select_phase(selected, phase)
    if not reused:
        inputs.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        batch.load_ranges(manifest_path)
        master.to_csv(master_path, index=False)
        provenance["annual_master_sha256"] = batch.file_sha256(master_path)
        (inputs / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    plot_provenance = {
        "period": period.key, "selection": selection.selection_metadata(master, args.min_occupied_blocks),
        "phases": phases, "master": str(master_path),
        "master_sha256": provenance["annual_master_sha256"],
        "source_provenance_sha256": batch.file_sha256(inputs / "provenance.json"),
        "plot_config": {"spatial": spatial.PLOT_CONFIG, "ae": ae.PLOT_CONFIG},
    }
    plot_provenance["code_sha256"] = {
        module.__file__: batch.file_sha256(Path(module.__file__))
        for module in (spatial, ae, statistics_phases, selection)
    }
    plot_provenance["code_sha256"][str(Path(__file__).resolve())] = batch.file_sha256(Path(__file__))
    output.mkdir(parents=True, exist_ok=True)
    (output / "provenance.json").write_text(json.dumps(
        plot_provenance, indent=2, default=lambda value: value.tolist()
    ) + "\n")
    print(f"Prepared {len(master)} phase rows: {output}", flush=True)
    for source in provenance["sources"]:
        print(f"{source['month']}: {source['status']} "
              f"{source.get('range_status_counts', source.get('reason'))}", flush=True)
    if not args.prepare_only:
        common = ["--ranges", str(manifest_path), "--master", str(master_path),
                  "--min-occupied-blocks", str(args.min_occupied_blocks)]
        for phase in phases:
            common.extend(["--phase", phase])
        spatial.main(common + ["--output-dir", str(output / "kappa_x0_statistics"),
                               "--min-bin-count", str(args.min_bin_count)])
        ae.main(common + ["--output-dir", str(output / "ae_statistics")])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
