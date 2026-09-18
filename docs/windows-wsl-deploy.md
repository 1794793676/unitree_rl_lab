# WSL 本地训练与 MuJoCo 部署（G1 29 自由度）

本文以当前 WSL 目录 `/home/dolphin/Unitree/unitree_rl_lab` 为准，替换原来的 Windows 训练 → GitHub → WSL 部署说明。保留文档文件名以兼容现有链接。

## 1. 训练环境与机器人资源

WSL 训练需要自己的 Linux Isaac Sim、Isaac Lab 和 Python 依赖，不能直接使用 Windows 的 Conda 环境。先激活已安装 Isaac Lab 的 WSL Python 环境，并按主 README 安装本项目；环境名称以本机实际配置为准。

当前目录布局：

```text
/home/dolphin/Unitree/
├── IsaacLab/
├── unitree_rl_lab/
├── unitree_ros/
├── unitree_mujoco/
└── unitree_sdk2/
```

`source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py` 中配置：

```python
UNITREE_ROS_DIR = "/home/dolphin/Unitree/unitree_ros"
```

G1 29 自由度训练使用：

```text
/home/dolphin/Unitree/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf
```

路径应指向包含 `robots` 子目录的仓库根目录，不需要重复两层 `unitree_ros`。README 的 `<...>` 是占位符；Python 路径字符串中的 `~` 不会自动展开，当前资源加载链路也没有展开它，因此使用绝对路径。换用户或移动资源目录后需相应修改。

## 2. 启动训练与模型保存位置

在已激活的 WSL Isaac Lab 环境中执行：

```bash
cd /home/dolphin/Unitree/unitree_rl_lab
python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Velocity
```

可用 `--num_envs` 调整并行环境数量，`--max_iterations` 调整训练迭代数，`--run_name` 添加本次训练的目录后缀。需要动作裁剪时显式传入 `--clip_actions 5.0`；当前随仓库保存的部署包记录了这一设置，但它不代表所有新训练的默认参数。

默认保存位置：

```text
/home/dolphin/Unitree/unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_velocity/
└── YYYY-MM-DD_HH-MM-SS[_run_name]/
    ├── model_100.pt
    ├── model_200.pt
    ├── ...
    └── params/
        ├── env.yaml
        ├── agent.yaml
        └── deploy.yaml
```

此任务默认每 100 次迭代保存检查点。日志根目录根据启动命令时的工作目录计算，因此先进入项目根目录。终端的 `Logging experiment in directory:` 会打印实际路径。不同任务或配置覆盖可能改变实验目录名和保存间隔。

`.pt` 检查点和完整训练日志被 Git 忽略，不会随 `git push` 上传。若要从 Windows 继续训练，需要另外复制所需的训练目录（含检查点和参数），并确认环境及配置兼容；只有 ONNX 部署包不足以恢复训练状态。

## 3. 导出部署模型

训练保存 `.pt`，不会自动替换控制器正在使用的 ONNX 模型。停止控制器后，选择一个检查点导出。以下占位符需替换为实际目录和迭代数：

```bash
cd /home/dolphin/Unitree/unitree_rl_lab
python scripts/rsl_rl/export_deploy.py \
  --checkpoint logs/rsl_rl/unitree_g1_29dof_velocity/<训练目录>/model_<迭代数>.pt
```

保留该检查点同一训练目录下的 `params/agent.yaml` 和 `params/deploy.yaml`，不要在导出前手动修改参数或观测顺序。

默认输出：

```text
deploy/robots/g1_29dof/config/policy/velocity/windows_trained/
├── exported/policy.onnx
├── params/deploy.yaml
└── provenance.json
```

`windows_trained` 是沿用的目录名，WSL 训练的模型同样可以写入。导出会更新这里的现有部署文件，G1 的 `config/config.yaml` 已通过 `FSM.Velocity.policy_dir` 指向此目录。上游原始策略仍在 `velocity/v0`，需要使用时应明确修改配置。

导出命令只使用 CPU，不启动 Isaac Sim；需要 PyTorch、RSL-RL 5.x、TensorDict、PyYAML、NumPy 和 ONNX。它重建包含观测归一化的 MLP actor，导出 batch 为 1 的 `obs → actions`，检查 ONNX 有效性，并用九组输入比较 PyTorch 与 ONNX 输出，通过后才写入部署包。

适用范围是当前 G1 29 自由度速度任务的单一 policy 观测组和 RSL-RL 5 `actor_state_dict` 检查点。旧检查点、循环网络、CNN 或自定义观测需要相应的导出支持。仓库原有 `play.py` 使用较旧的 RSL-RL API，不能代替这里针对 RSL-RL 5 的导出命令。

截至本文更新，仓库部署包的 `provenance.json` 记录来源为 `2026-09-16_09-43-25_resume11700_actionclip5/model_19600.pt`，RSL-RL 版本为 `5.0.1`。每次重新导出后以该文件为准。数值一致性验证不代表步行性能验证；当前记录的 `mujoco_tested` 为 `false`。

## 4. MuJoCo 与控制器准备

按主 README 安装 Linux 编译依赖、Unitree SDK2（安装至 `/usr/local`）和 `unitree_mujoco`。编译 G1 控制器：

```bash
cd /home/dolphin/Unitree/unitree_rl_lab/deploy/robots/g1_29dof
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
```

仅运行 C++ 部署推理不需要 Isaac Lab 或训练用 URDF；在 WSL 训练则需要第 1 节的训练环境。

在 `/home/dolphin/Unitree/unitree_mujoco/simulate/config.yaml` 中设置：

```yaml
robot: "g1"
robot_scene: "scene_29dof.xml"
domain_id: 0
interface: "lo"
enable_elastic_band: 1
use_joystick: 0
joystick_type: "xbox"
joystick_device: "/dev/input/js0"
```

控制器的 `config/config.yaml` 中，`FSM.Velocity.keyboard_control: true` 启用键盘速度指令。配合 `use_joystick: 0` 无需手柄。若使用手柄，将上述两项分别改为 `false` 和 `1`；通过 Windows USB 转发时，确认 WSL 中 `/dev/input/js0` 可读。

## 5. 启动与键盘操作

终端 A：

```bash
cd /home/dolphin/Unitree/unitree_mujoco/simulate/build
./unitree_mujoco
```

终端 B：

```bash
cd /home/dolphin/Unitree/unitree_rl_lab/deploy/robots/g1_29dof/build
./g1_ctrl --network lo
```

确认 `Policy directory:` 日志以 `velocity/windows_trained` 结尾，并等待 `Connected to robot.`。

1. 在控制器终端按 `1` 进入 FixStand，等待 3 秒完成站立过渡。
2. 切换焦点到 MuJoCo 窗口，按 `8` 调低悬挂高度，直到双脚接触地面。
3. 回到控制器终端按 `2` 进入 Velocity。
4. 在 MuJoCo 窗口按 `9` 释放弹力带。

以下按键在**控制器终端**输入，字母使用小写：

| 按键 | 功能 |
| --- | --- |
| `1` | Passive → FixStand |
| `2` | FixStand → Velocity，不能从 Passive 直接进入 |
| `0` | FixStand 或 Velocity → Passive |
| `w` / `s` | 前进 / 后退 |
| `a` / `d` | 左移 / 右移 |
| `q` / `e` | 左转 / 右转 |

移动按键产生单位指令，再限制到策略配置的速度范围。范围以当前部署参数为准。约 80 毫秒未收到终端按键事件后，速度指令归零，机器人实际停止可能有延迟。长按依赖操作系统按键重复，首次重复前可能出现停顿；不支持多个移动键同时输入，未映射按键也会使指令归零。手柄状态切换组合键仍可用：LT + Up、RB + X、LT + B。

更新模型前先停止控制器，导出后重启。仅更新模型或 YAML 无需重新编译；修改 C++ 后需重新构建。

## 6. 验证命令

在已安装导出依赖的 WSL Python 环境中运行 CPU 导出回归检查：

```bash
cd /home/dolphin/Unitree/unitree_rl_lab
python -m unittest discover -s scripts/rsl_rl/tests -p test_export_deploy.py
```

检查覆盖归一化 actor 数值一致性、参数复制、观测维度不匹配和旧检查点拒绝逻辑，不替代 MuJoCo 行为测试。

键盘回归检查使用伪终端，无需手柄或仿真器：

```bash
cmake -S deploy/robots/g1_29dof -B deploy/robots/g1_29dof/build -DBUILD_TESTING=ON
cmake --build deploy/robots/g1_29dof/build -j4
ctest --test-dir deploy/robots/g1_29dof/build --output-on-failure
```

## 7. 与原 Windows 流程的区别及可选同步

| 项目 | 原 Windows → WSL 流程 | 当前 WSL 本地流程 |
| --- | --- | --- |
| 训练环境 | Windows Isaac Lab / Python | WSL Linux Isaac Lab / Python |
| 机器人资源 | Windows 本机路径 | `/home/dolphin/Unitree/unitree_ros` |
| 检查点 | Windows 项目下的 `logs/rsl_rl/` | WSL 项目下的同名相对目录 |
| 部署交接 | Windows 导出 → GitHub → WSL 拉取 | 本地导出后直接重启控制器 |
| 部署格式与目录 | ONNX、配套 YAML、来源记录 | 相同，沿用 `windows_trained` 目录 |

本机训练和部署无需通过 GitHub 中转。需要备份或跨机器同步部署包时，在导出成功后一起提交三个部署文件：

```bash
cd /home/dolphin/Unitree/unitree_rl_lab
git add deploy/robots/g1_29dof/config/policy/velocity/windows_trained
git diff --cached --stat
git commit -m "Update G1 deployment policy"
git push origin main
```

仓库 `origin` 为 `https://github.com/1794793676/unitree_rl_lab.git`，`upstream` 为 `https://github.com/unitreerobotics/unitree_rl_lab.git`。分支名按实际情况调整。代码和配置若同时变化，应明确加入对应文件，不要使用 `git add .` 混入无关改动。

其他机器接收更新前先停止控制器，处理好本地未提交改动，然后在同一分支执行 `git pull --ff-only origin main`。若分支已分叉，先处理历史差异；不要在导出过程中推送或拉取。
