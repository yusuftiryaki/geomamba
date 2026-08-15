import numpy as np

from geomamba.data import faults, lorentz_particle, lorenz, oscillators, rigid_body
from geomamba.train import anomaly, physics


def test_shapes():
    assert oscillators.generate(3, 20, seed=0)["x"].shape == (3, 20, 4)
    assert lorenz.generate(2, 20, seed=0)["x"].shape == (2, 20, 3)
    out = rigid_body.generate(3, 20, seed=0)
    assert out["x"].shape == (3, 20, 3) and out["params"].shape == (3, 3)
    out = lorentz_particle.generate(3, 20, seed=0)
    assert out["x"].shape == (3, 20, 3) and out["params"].shape == (3, 3)


def test_rigid_body_ground_truth_conserves_invariants():
    out = rigid_body.generate(4, 200, seed=1)
    e = rigid_body.energy(out["x"].astype(np.float64), out["params"].astype(np.float64))
    l = rigid_body.momentum_norm(out["x"].astype(np.float64), out["params"].astype(np.float64))
    assert physics.relative_drift(e)[:, -1].max() < 1e-4
    assert physics.relative_drift(l)[:, -1].max() < 1e-4


def test_particle_ground_truth_conserves_speed():
    out = lorentz_particle.generate(4, 200, seed=1)
    s = lorentz_particle.speed(out["x"].astype(np.float64))
    assert physics.relative_drift(s)[:, -1].max() < 1e-4


def test_anomalous_generation():
    for fn in faults.ANOMALOUS.values():
        out = fn(20, 60, seed=0, anomaly_frac=0.5)
        assert out["x"].shape == (20, 60, 3)
        anom = out["labels"].max(axis=1) > 0.5
        assert 0 < anom.sum() < 20
        # labels are contiguous suffixes
        for lab in out["labels"][anom]:
            tau = int(np.argmax(lab > 0.5))
            assert (lab[tau:] == 1).all() and (lab[:tau] == 0).all()


def test_auroc_sanity():
    scores = np.array([[0.1, 0.2, 0.9, 0.95], [0.1, 0.15, 0.12, 0.11]])
    labels = np.array([[0, 0, 1, 1], [0, 0, 0, 0]])
    assert anomaly.auroc(scores, labels) == 1.0
    assert anomaly.auroc(-scores, labels) == 0.0
