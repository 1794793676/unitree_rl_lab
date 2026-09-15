# Windows training → GitHub → WSL MuJoCo

## Repository layout

- `origin`: https://github.com/1794793676/unitree_rl_lab.git
- `upstream`: https://github.com/unitreerobotics/unitree_rl_lab.git
- Windows trains in its existing Isaac Lab environment.
- WSL runs `deploy/robots/g1_29dof/build/g1_ctrl` and `unitree_mujoco`.
- Git tracks the selected ONNX model, its matching deployment parameters and provenance:

```text
deploy/robots/g1_29dof/config/policy/velocity/windows_trained/
├── exported/policy.onnx
├── params/deploy.yaml
└── provenance.json
```

The G1 `config/config.yaml` selects this exact directory. Build outputs, full
training logs and `.pt` checkpoints remain ignored. The original upstream policy
remains in `velocity/v0`; select that exact directory to use it again.

The initial Windows bundle is from `2026-09-11_10-55-35/model_100.pt` (iteration
100). Its exported outputs were checked against the PyTorch actor on CPU. This is
an early training checkpoint, not a claim of stable walking. MuJoCo behavior has
not been verified. See `provenance.json` for hashes and verification results.

## Windows: export a selected checkpoint

PowerShell, with the existing `isaac` Conda environment:

```powershell
conda activate isaac
cd C:\Users\Dolphin\Desktop\Unitree\unitree_rl_lab
python scripts/rsl_rl/export_deploy.py --checkpoint logs/rsl_rl/unitree_g1_29dof_velocity/2026-09-11_10-55-35/model_100.pt
```

Replace the checkpoint path after training a new policy. Keep `params/agent.yaml`
and `params/deploy.yaml` beside that checkpoint in its original training run.
Do not edit or reorder the training parameters before export.

This CPU-only command requires the existing PyTorch, RSL-RL 5.x, TensorDict,
PyYAML, NumPy and ONNX packages; it does not launch Isaac Sim. It reconstructs the
RSL-RL MLP actor including observation normalization, exports static batch-one
`obs → actions`, checks ONNX validity, compares nine inputs with the PyTorch
actor, and publishes the three deployment files only after verification passes.

Scope: the existing G1 29-DoF velocity observation/action layout with a single
policy observation group and an RSL-RL 5 `actor_state_dict` checkpoint. Legacy
checkpoints, recurrent/CNN models and custom observations need their own export
path. The repository's original `play.py` uses older RSL-RL APIs and is not the
export command for this installed RSL-RL 5 environment.

Export is not a walking-performance test. Check the learned behavior in simulation
before treating a checkpoint as a usable controller.

## Windows: publish the bundle

After the export command succeeds, review and commit the bundle together:

```powershell
git add deploy/robots/g1_29dof/config/policy/velocity/windows_trained
git diff --cached --stat
git commit -m "Update G1 Windows-trained deployment policy"
git push origin main
```

If you also change C++ code, the exporter or deployment configuration, explicitly
add those files to the same update. Do not use `git add .` to accidentally include
unrelated local work. Do not push or pull halfway through an export.

## WSL: first setup

For a fresh checkout:

```bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/1794793676/unitree_rl_lab.git
cd unitree_rl_lab
```

For an existing Git checkout with a clean working tree:

```bash
cd ~/projects/unitree_rl_lab
git status
git remote set-url origin https://github.com/1794793676/unitree_rl_lab.git
git fetch origin
git switch main
git pull --ff-only origin main
```

Commit or stash local changes first if needed; do not overwrite them. If `--ff-only`
reports divergent history, reconcile the branches before deployment.

Install Unitree SDK2 into `/usr/local`, Linux build dependencies and
`unitree_mujoco` as described in the repository's main README. Then build once:

```bash
cd ~/projects/unitree_rl_lab/deploy/robots/g1_29dof
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
```

No Windows Python environment, URDF training asset path or Isaac Lab installation
is required for this C++ deployment. WSL needs its own Linux libraries and build.

In `~/projects/unitree_mujoco/simulate/config.yaml`, set:

```yaml
robot: "g1"
robot_scene: "scene_29dof.xml"
domain_id: 0
interface: "lo"
enable_elastic_band: 1
use_joystick: 1
joystick_type: "xbox"
joystick_device: "/dev/input/js0"
```

The current controller uses a joystick for state changes. The earlier proposed
keyboard state-switching feature is not implemented by this repository-sync
change. `use_joystick: 0` alone does not let you start the policy without a gamepad.
With Windows USB passthrough, check that WSL exposes a readable `/dev/input/js0`.

## WSL: subsequent updates and inference

Stop the controller before updating. Pull from the same branch used on Windows:

```bash
cd ~/projects/unitree_rl_lab
git pull --ff-only origin main
```

Model/YAML-only updates require a restart, not compilation. For C++ changes:

```bash
cmake --build deploy/robots/g1_29dof/build -j4
```

Terminal A:

```bash
cd ~/projects/unitree_mujoco/simulate/build
./unitree_mujoco
```

Terminal B:

```bash
cd ~/projects/unitree_rl_lab/deploy/robots/g1_29dof/build
./g1_ctrl --network lo
```

Check the `Policy directory:` log ends in `velocity/windows_trained` and wait for
`Connected to robot.` Then use LT + Up to stand; wait for the transition to finish;
press 8 in the MuJoCo window until the feet touch the ground; use RB + X to start
the policy; press 9 in MuJoCo to release the elastic band.

## Export verification

In the Windows Isaac environment:

```powershell
python -m unittest discover -s scripts/rsl_rl/tests -p test_export_deploy.py
```

These CPU checks cover normalized-actor numerical equivalence, matching parameter
copying, rejection of mismatched observation dimensions and legacy checkpoints.
They do not replace running the controller in WSL/MuJoCo.
