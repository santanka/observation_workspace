"""Archive pre-67-eV results on the same filesystem; retain old path references.

Default is read-only. --apply renames the two explicit directories and leaves
compatibility symlinks. Raw observations and range manifests are untouched.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone

ROOT = Path("/mnt/j/statistical_analysis_arase/preanalysis/KAW_observation")
ARCHIVE = ROOT / "legacy" / "lepe_unrestricted_20261006"
NAMES = ("auto", "run_state")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    # Do not relocate results while an analysis or figure process uses them.
    processes = subprocess.check_output(["ps", "-eo", "comm=,args="], text=True)
    for line in processes.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) == 2 and fields[0].startswith("python"):
            # Standalone interactive kernels do not use the batch auto/state
            # directories. Batch kernels have a runner parent caught here.
            if "batch_arase.py" in fields[1] or "statistical_analysis_arase_pre_auto.batch_arase" in fields[1] or "statistical_analysis_arase_pre_auto/plot_" in fields[1]:
                raise RuntimeError(f"Active analysis/plot process: {line.strip()}")
    plan = []
    for name in NAMES:
        source, target = ROOT / name, ARCHIVE / name
        if source.is_symlink():
            if source.resolve() != target or not target.is_dir():
                raise RuntimeError(f"Unexpected existing symlink: {source}")
            state = "already_archived"
        else:
            if not source.is_dir() or target.exists():
                raise RuntimeError(f"Unexpected source/destination: {source} -> {target}")
            state = "pending"
        plan.append({"source": str(source), "archive": str(target), "state": state})
    print(json.dumps(plan, indent=2))
    if not args.apply:
        return
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    for entry in plan:
        if entry["state"] == "already_archived":
            continue
        source, target = Path(entry["source"]), Path(entry["archive"])
        os.rename(source, target)
        try:
            source.symlink_to(target, target_is_directory=True)
        except Exception:
            os.rename(target, source)
            raise
        print(f"Archived {source} -> {target}; compatibility symlink installed")
    manifest = ARCHIVE / "migration.json"
    if not manifest.exists():
        manifest.write_text(json.dumps({
            "migrated_at": datetime.now(timezone.utc).isoformat(),
            "old_lepe_energy_policy": "no_explicit_energy_cut",
            "new_analysis_version": "lepe_ge67eV_v1",
            "paths": plan,
            "raw_observations_moved": False,
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
