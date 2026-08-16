"""Serial experiment grid runner — works locally and on Kaggle/Colab (GPU
auto-detected by the trainer). Generates data, trains, evaluates, aggregates.

Usage:
  python scripts/run_grid.py --grid configs/grid/m3_rigid_body.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


SRC = str(Path(__file__).resolve().parent.parent / "src")


def sh(args: list[str]) -> None:
    print("+", " ".join(args), flush=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(args, check=True, env=env)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True, type=Path)
    ap.add_argument("--data-only", action="store_true",
                    help="generate datasets and exit (run once before sharding)")
    ap.add_argument("--skip-data", action="store_true")
    ap.add_argument("--shard", default=None,
                    help="'k/n': run only every n-th run starting at k, for "
                         "one-process-per-GPU parallelism")
    args = ap.parse_args()
    grid = yaml.safe_load(args.grid.read_text())
    shard_k, shard_n = 0, 1
    if args.shard:
        shard_k, shard_n = (int(v) for v in args.shard.split("/"))

    for ds in [] if args.skip_data else grid.get("datasets", []):
        out = Path(ds["out"])
        if out.exists():
            print(f"skip existing {out}")
            continue
        cmd = [sys.executable, "scripts/generate_data.py",
               "--system", ds["system"], "--out", str(out),
               "--length", str(ds.get("length", 257))]
        if ds.get("anomalous"):
            cmd += ["--anomalous", "--n-test", str(ds.get("n_test", 500))]
        else:
            cmd += ["--n-train", str(ds.get("n_train", 3000)),
                    "--n-val", str(ds.get("n_val", 500)),
                    "--n-test", str(ds.get("n_test", 500))]
        sh(cmd)

    if args.data_only:
        return

    suffix = f"-shard{shard_k}" if shard_n > 1 else ""
    rows = []
    for run in grid["runs"][shard_k::shard_n]:
        overrides = [str(o) for o in run.get("overrides", [])]
        name = run.get("name") or Path(run["config"]).stem
        sh([sys.executable, "scripts/train.py", "--config", run["config"],
            f"name={name}", *overrides])
        run_dir = sorted(Path("experiments/runs").glob(f"{name}-*"))[-1]
        eval_cmd = [sys.executable, "scripts/evaluate.py", "--run", str(run_dir)]
        if run.get("anom"):
            eval_cmd += ["--anom", run["anom"]]
        eval_cmd += [str(a) for a in run.get("eval_args", [])]
        sh(eval_cmd)
        row = json.loads((run_dir / "eval.json").read_text())
        row["name"] = name
        row["overrides"] = " ".join(overrides)
        rows.append(row)
        # incremental aggregate so partial grids are still usable, mirrored to
        # the Kaggle output dir so a session timeout cannot lose finished runs
        payload = json.dumps(rows, indent=2)
        Path(f"experiments/grid_results{suffix}.json").write_text(payload)
        for mirror in (Path("/kaggle/working"),):
            if mirror.is_dir():
                (mirror / f"grid_results{suffix}.json").write_text(payload)
                try:
                    import pandas as pd
                    pd.DataFrame(rows).to_csv(
                        mirror / f"grid_results{suffix}.csv", index=False)
                except ImportError:
                    pass

    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        df.to_csv(f"experiments/grid_results{suffix}.csv", index=False)
        cols = [c for c in ["name", "overrides", "test_mse", "rollout_mse_h100",
                            "energy_drift", "momentum_drift", "anomaly_auroc_2s"]
                if c in df.columns]
        print(df[cols].to_string(index=False))
    except ImportError:
        pass


if __name__ == "__main__":
    main()
