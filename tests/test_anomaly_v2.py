import numpy as np
import torch

from geomamba.data import rigid_body, faults
from geomamba.train import anomaly as A


def test_cusum_detects_persistent_small_shift():
    rng = np.random.default_rng(0)
    scores = rng.lognormal(0, 0.3, size=(20, 200))
    scores[10:, 120:] *= 1.6  # small persistent shift, invisible per-step
    labels = np.zeros((20, 200)); labels[10:, 120:] = 1
    raw_auroc = A.auroc(scores, labels)
    cusum_auroc = A.auroc(A.cusum_scores(scores), labels)
    assert cusum_auroc > raw_auroc + 0.1
    assert cusum_auroc > 0.85


def test_invariant_scores_flag_inertia_jump():
    normal = rigid_body.generate(8, 200, seed=0)["x"]
    anom = faults.rigid_body_anomalous(40, 200, seed=1, anomaly_frac=0.5)
    scores = A.invariant_scores(anom["x"])
    labels = anom["labels"][:, 1:]
    assert A.auroc(scores, labels) > 0.8
    # healthy trajectories stay on their level sets
    s_norm = A.invariant_scores(normal)
    assert np.median(s_norm) < 3.0


def test_event_f1_basics():
    scores = np.zeros((6, 100)); labels = np.zeros((6, 100))
    scores[0, 50:] = 10.0; labels[0, 48:] = 1     # detected on time
    labels[1, 60:] = 1                             # missed
    scores[2, 30:33] = 10.0                        # brief false alarm on a normal traj
    # trajs 3-5: clean normals defining the threshold
    m = A.event_f1(scores, labels, tolerance=20)
    assert m["event_recall"] == 0.5 and m["event_precision"] == 0.5
