"""Export a G1 velocity RSL-RL 5 MLP checkpoint to the Git-tracked WSL bundle.

Runs on CPU without Isaac Sim. Uses the installed RSL-RL actor/export wrapper,
including its observation normalizer, and checks ONNX outputs before publishing.
"""

import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import torch
import yaml
from rsl_rl.models import MLPModel
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "deploy/robots/g1_29dof/config/policy/velocity/windows_trained"


class ClippedOnnxActor(torch.nn.Module):
    """Match RslRlVecEnvWrapper clipping before action scaling and last_action."""

    def __init__(self, actor, limit):
        super().__init__()
        self.actor = actor
        self.limit = limit

    def forward(self, obs):
        return torch.clamp(self.actor(obs), -self.limit, self.limit)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_checkpoint(checkpoint, output=DEFAULT_OUTPUT):
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    params = checkpoint.parent / "params"
    agent = yaml.safe_load((params / "agent.yaml").read_text(encoding="utf-8"))
    deploy = yaml.safe_load((params / "deploy.yaml").read_text(encoding="utf-8"))
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "actor_state_dict" not in saved:
        raise ValueError("Expected RSL-RL 5 actor_state_dict; use the matching legacy play.py for older checkpoints.")
    actor_cfg = copy.deepcopy(agent.get("actor", {}))
    if actor_cfg.pop("class_name", None) != "MLPModel":
        raise ValueError("Only RSL-RL 5 MLPModel G1 velocity actors are supported.")
    version = importlib.metadata.version("rsl-rl-lib")
    if version.split(".")[0] != "5":
        raise ValueError(f"This exporter requires RSL-RL 5.x; installed version is {version}.")
    groups = agent.get("obs_groups", {})
    if groups and groups.get("actor") != ["policy"]:
        raise ValueError("Only the single policy observation group is supported.")
    clip_actions = agent.get("clip_actions")
    if clip_actions is not None:
        if type(clip_actions) not in (int, float) or not (0.0 < clip_actions < float("inf")):
            raise ValueError("clip_actions must be a finite positive number or null.")

    # Restrict this utility to the existing G1 velocity deploy contract.
    dimensions = dict(base_ang_vel=3, projected_gravity=3, velocity_commands=3,
                      joint_pos_rel=29, joint_vel_rel=29, last_action=29)
    observations = deploy["observations"]
    if list(observations) != list(dimensions):
        raise ValueError("Expected the existing G1 velocity observation terms in their original order.")
    if sorted(deploy["joint_ids_map"]) != list(range(29)):
        raise ValueError("Expected a 29-joint G1 mapping.")
    actions = deploy["actions"]
    if list(actions) != ["JointPositionAction"] or len(actions["JointPositionAction"]["scale"]) != 29:
        raise ValueError("Expected 29 JointPositionAction outputs.")
    for name, dim in dimensions.items():
        if len(observations[name]["scale"]) != dim or observations[name]["history_length"] < 1:
            raise ValueError(f"Invalid observation scale/history for {name}.")
    input_dim = sum(dim * observations[name]["history_length"] for name, dim in dimensions.items())
    state = saved["actor_state_dict"]
    if state["mlp.0.weight"].shape[1] != input_dim:
        raise ValueError("Checkpoint observation dimension does not match params/deploy.yaml.")
    actor = MLPModel(TensorDict({"policy": torch.zeros(1, input_dim)}, [1]),
                     {"actor": ["policy"]}, "actor", 29, **actor_cfg)
    actor.load_state_dict(state, strict=True)
    actor.eval()
    onnx_actor = actor.as_onnx(verbose=False).cpu().eval()
    dummy_inputs = onnx_actor.get_dummy_inputs()
    if clip_actions is not None:
        onnx_actor = ClippedOnnxActor(onnx_actor, clip_actions).eval()

    # Validate the entire bundle before touching an existing deployment.
    with tempfile.TemporaryDirectory(prefix="unitree-export-") as staging:
        model_path = Path(staging) / "policy.onnx"
        deploy_path = Path(staging) / "deploy.yaml"
        deploy_path.write_text((params / "deploy.yaml").read_text(encoding="utf-8"),
                               encoding="utf-8", newline="\n")
        torch.onnx.export(onnx_actor, dummy_inputs, str(model_path),
                          input_names=["obs"], output_names=["actions"],
                          opset_version=18, dynamo=False)
        model = onnx.load(str(model_path))
        onnx.checker.check_model(model, full_check=True)
        evaluator = ReferenceEvaluator(model)
        generator = torch.Generator().manual_seed(42)
        max_error = 0.0
        samples = [torch.zeros(1, input_dim)] + [torch.randn(1, input_dim, generator=generator) for _ in range(8)]
        with torch.inference_mode():
            for obs in samples:
                expected = actor(TensorDict({"policy": obs}, [1])).numpy()
                if clip_actions is not None:
                    expected = np.clip(expected, -clip_actions, clip_actions)
                actual = evaluator.run(None, {"obs": obs.numpy()})[0]
                if not np.isfinite(actual).all() or not np.isfinite(expected).all():
                    raise ValueError("Non-finite policy outputs.")
                np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
                max_error = max(max_error, float(np.max(np.abs(actual - expected))))
        manifest = {
            "task": "Unitree-G1-29dof-Velocity",
            "training_run": checkpoint.parent.name,
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": sha256(checkpoint),
            "iteration": saved.get("iter"),
            "rsl_rl_version": version,
            "torch_version": torch.__version__,
            "clip_actions": clip_actions,
            "input_name": "obs", "input_shape": [1, input_dim],
            "output_name": "actions", "output_shape": [1, 29],
            "onnx_sha256": sha256(model_path),
            "deploy_yaml_sha256": sha256(deploy_path),
            "validation": {"backend": "onnx.reference.ReferenceEvaluator", "samples": len(samples),
                           "max_absolute_error": max_error, "mujoco_tested": False},
        }
        (output / "exported").mkdir(parents=True, exist_ok=True)
        (output / "params").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(model_path, output / "exported/policy.onnx")
        shutil.copyfile(deploy_path, output / "params/deploy.yaml")
        (output / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Exported and numerically verified: {output}")
    print(f"Checkpoint iteration: {saved.get('iter')}; max absolute error: {max_error:.3g}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    export_checkpoint(args.checkpoint, args.output)
