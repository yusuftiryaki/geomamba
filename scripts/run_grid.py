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
    args = ap.parse_args()
    grid = yaml.safe_load(args.grid.read_text())

    for ds in grid.get("datasets", []):
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

    rows = []
    for run in grid["runs"]:
        overrides = [str(o) for o in run.get("overrides", [])]
        name = run.get("name") or Path(run["config"]).stem
        sh([sys.executable, "scripts/train.py", "--config", run["config"],
            f"name={name}", *overrides])
        run_dir = sorted(Path("experiments/runs").glob(f"{name}-*"))[-1]
        eval_cmd = [sys.executable, "scripts/evaluate.py", "--run", str(run_dir)]
        if run.get("anom"):
            eval_cmd += ["--anom", run["anom"]]
        sh(eval_cmd)
        row = json.loads((run_dir / "eval.json").read_text())
        row["name"] = name
        row["overrides"] = " ".join(overrides)
        rows.append(row)
        # incremental aggregate so partial grids are still usable
        agg = Path("experiments/grid_results.json")
        agg.write_text(json.dumps(rows, indent=2))

    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        df.to_csv("experiments/grid_results.csv", index=False)
        cols = [c for c in ["name", "overrides", "test_mse", "rollout_mse_h100",
                            "energy_drift", "momentum_drift", "anomaly_auroc_2s"]
                if c in df.columns]
        print(df[cols].to_string(index=False))
    except ImportError:
        pass


if __name__ == "__main__":
    main()
