"""Portable intermediate policy snapshots for the public RBC trainers."""

import os
import tempfile
import unittest
from pathlib import Path

from jax import numpy as jnp

from DEQN.econ_models.RBC.train_shared import (
    SHARED_RBC_TRAIN,
    load_policy_params,
    maybe_save_policy_snapshot,
    policy_snapshot_path,
    resolve_policy_snapshot_epochs,
    save_policy_params,
)


class ResolvePolicySnapshotEpochsTest(unittest.TestCase):
    def test_default_config_is_off(self):
        self.assertEqual(resolve_policy_snapshot_epochs(SHARED_RBC_TRAIN), frozenset())
        self.assertEqual(resolve_policy_snapshot_epochs({"n_epochs": 20}), frozenset())
        self.assertEqual(
            resolve_policy_snapshot_epochs({"n_epochs": 20, "policy_snapshot_every_n_epochs": None}),
            frozenset(),
        )
        self.assertEqual(
            resolve_policy_snapshot_epochs({"n_epochs": 20, "policy_snapshot_every_n_epochs": 0}),
            frozenset(),
        )
        self.assertEqual(
            resolve_policy_snapshot_epochs({"n_epochs": 20, "policy_snapshot_epochs": []}),
            frozenset(),
        )

    def test_every_n_includes_epoch_zero_and_final(self):
        self.assertEqual(
            resolve_policy_snapshot_epochs(
                {"n_epochs": 20, "policy_snapshot_every_n_epochs": 5}
            ),
            frozenset({0, 5, 10, 15, 20}),
        )
        self.assertEqual(
            resolve_policy_snapshot_epochs(
                {"n_epochs": 22, "policy_snapshot_every_n_epochs": 5}
            ),
            frozenset({0, 5, 10, 15, 20, 22}),
        )
        self.assertEqual(
            resolve_policy_snapshot_epochs(
                {"n_epochs": 3, "policy_snapshot_every_n_epochs": 10}
            ),
            frozenset({0, 3}),
        )

    def test_explicit_epochs_union_zero_and_final(self):
        self.assertEqual(
            resolve_policy_snapshot_epochs(
                {"n_epochs": 20, "policy_snapshot_epochs": [10]}
            ),
            frozenset({0, 10, 20}),
        )
        self.assertEqual(
            resolve_policy_snapshot_epochs(
                {"n_epochs": 8, "policy_snapshot_epochs": 4}
            ),
            frozenset({0, 4, 8}),
        )

    def test_explicit_and_every_n_are_unioned_and_clipped(self):
        self.assertEqual(
            resolve_policy_snapshot_epochs(
                {
                    "n_epochs": 10,
                    "policy_snapshot_every_n_epochs": 4,
                    "policy_snapshot_epochs": [3, 99, -1],
                }
            ),
            frozenset({0, 3, 4, 8, 10}),
        )


class PolicySnapshotRoundtripTest(unittest.TestCase):
    def test_snapshot_path_and_msgpack_roundtrip(self):
        params = {"w": jnp.array([1.0, 2.0])}
        with tempfile.TemporaryDirectory() as run_dir:
            path = maybe_save_policy_snapshot(params, run_dir, 0, {0, 2})
            self.assertEqual(path, policy_snapshot_path(run_dir, 0))
            self.assertEqual(os.path.basename(path), "params_epoch_0000.msgpack")
            self.assertIsNone(maybe_save_policy_snapshot(params, run_dir, 1, {0, 2}))
            self.assertFalse(os.path.exists(policy_snapshot_path(run_dir, 1)))

            loaded = load_policy_params(params, path)
            self.assertTrue(jnp.allclose(loaded["w"], params["w"]))

            final_path = maybe_save_policy_snapshot(params, Path(run_dir), 2, {0, 2})
            save_policy_params(params, os.path.join(run_dir, "params.msgpack"))
            self.assertTrue(os.path.exists(final_path))
            self.assertTrue(os.path.exists(os.path.join(run_dir, "params.msgpack")))


if __name__ == "__main__":
    unittest.main()
