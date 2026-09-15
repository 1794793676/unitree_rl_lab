"""CPU integration checks for the Windows-to-WSL deployment bundle."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import yaml
from onnx.reference import ReferenceEvaluator
from rsl_rl.models import MLPModel
from tensordict import TensorDict

SCRIPT = Path(__file__).resolve().parents[1] / "export_deploy.py"
spec = importlib.util.spec_from_file_location("export_deploy", SCRIPT)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)
ROOT = SCRIPT.parents[2]


class ExportDeployTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name) / "run"
        (self.run / "params").mkdir(parents=True)
        self.output = Path(self.temp.name) / "bundle"
        self.deploy = yaml.safe_load((ROOT / "deploy/robots/g1_29dof/config/policy/velocity/v0/params/deploy.yaml").read_text())
        self.actor_cfg = dict(class_name="MLPModel", hidden_dims=[8, 4], activation="elu",
                              obs_normalization=True, distribution_cfg=None)
        self.actor = MLPModel(TensorDict({"policy": torch.zeros(1, 480)}, [1]),
                              {"actor": ["policy"]}, "actor", 29,
                              **{k: v for k, v in self.actor_cfg.items() if k != "class_name"})
        self.actor.update_normalization(TensorDict({"policy": torch.randn(32, 480) + 2}, [32]))
        self.actor.eval()
        (self.run / "params/agent.yaml").write_text(yaml.safe_dump({"actor": self.actor_cfg}))
        self.write_deploy()
        self.checkpoint = self.run / "model_100.pt"
        torch.save({"actor_state_dict": self.actor.state_dict(), "iter": 100}, self.checkpoint)

    def write_deploy(self):
        (self.run / "params/deploy.yaml").write_text(yaml.safe_dump(self.deploy, sort_keys=False))

    def test_export_preserves_normalized_actor_and_matching_config(self):
        exporter.export_checkpoint(self.checkpoint, self.output)
        obs = torch.randn(1, 480)
        with torch.inference_mode():
            expected = self.actor(TensorDict({"policy": obs}, [1])).numpy()
        actual = ReferenceEvaluator(str(self.output / "exported/policy.onnx")).run(None, {"obs": obs.numpy()})[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
        self.assertEqual((self.output / "params/deploy.yaml").read_text(),
                         (self.run / "params/deploy.yaml").read_text())
        self.assertNotIn(b"\r", (self.output / "params/deploy.yaml").read_bytes())
        manifest = json.loads((self.output / "provenance.json").read_text())
        self.assertEqual(manifest["iteration"], 100)
        self.assertEqual(manifest["input_shape"], [1, 480])

    def test_mismatched_config_does_not_replace_existing_model(self):
        self.output.mkdir()
        marker = self.output / "provenance.json"
        marker.write_text("existing bundle")
        self.deploy["observations"]["base_ang_vel"]["history_length"] = 1
        self.write_deploy()
        with self.assertRaisesRegex(ValueError, "observation dimension"):
            exporter.export_checkpoint(self.checkpoint, self.output)
        self.assertEqual(marker.read_text(), "existing bundle")

    def test_legacy_checkpoint_has_actionable_error(self):
        torch.save({"model_state_dict": {}}, self.checkpoint)
        with self.assertRaisesRegex(ValueError, "actor_state_dict"):
            exporter.export_checkpoint(self.checkpoint, self.output)


if __name__ == "__main__":
    unittest.main()
