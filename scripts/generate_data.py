"""Generate synthetic datasets as npz files with train/val/test splits.

Usage:
  python scripts/generate_data.py --system oscillators --out data/oscillators.npz \
      --n-train 3000 --n-val 500 --n-test 500 --length 513
  python scripts/generate_data.py --smoke   # tiny files for smoke tests
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from geomamba.data import faults, lorentz_particle, lorenz, oscillators, rigid_body

SYSTEMS = {
    "oscillators": oscillators.generate,
    "lorenz": lorenz.generate,
    "rigid_body": rigid_body.generate,
    "lorentz_particle": lorentz_particle.generate,
}


def build(system: str, out: Path, n_train: int, n_val: int, n_test: int,
          length: int, seed: int) -> None:
    gen = SYSTEMS[system]
    splits = {}
    for i, (name, n) in enumerate([("train", n_train), ("val", n_val), ("test", n_test)]):
        result = gen(n, length, seed=seed + 1000 * i)
        splits[name] = result["x"]
        if "params" in result:
            splits[f"{name}_params"] = result["params"]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **splits)
    print(f"{out}: " + ", ".join(f"{k}={v.shape}" for k, v in splits.items()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=SYSTEMS)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--length", type=int, default=513)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny oscillator dataset for smoke tests")
    ap.add_argument("--anomalous", action="store_true",
                    help="generate an anomalous test set (x + labels) instead")
    args = ap.parse_args()

    if args.anomalous:
        if args.system not in faults.ANOMALOUS or not args.out:
            ap.error("--anomalous needs --system rigid_body|lorentz_particle and --out")
        result = faults.ANOMALOUS[args.system](args.n_test, args.length,
                                               seed=args.seed + 9999)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out, **result)
        n_anom = int((result["labels"].max(axis=1) > 0.5).sum())
        print(f"{args.out}: x={result['x'].shape}, anomalous={n_anom}/{len(result['x'])}")
        return

    if args.smoke:
        build("oscillators", Path("data/smoke_oscillators.npz"),
              64, 16, 16, 193, seed=0)
        return
    if not args.system or not args.out:
        ap.error("--system and --out are required (or --smoke)")
    build(args.system, args.out, args.n_train, args.n_val, args.n_test,
          args.length, args.seed)


if __name__ == "__main__":
    main()
