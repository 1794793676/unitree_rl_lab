#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <unordered_map>

namespace isaaclab
{
// Selected by FSM.Velocity.keyboard_control in the controller config.
REGISTER_OBSERVATION(keyboard_velocity_commands)
{
    std::string key = FSMState::keyboard->key();
    const auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    static std::unordered_map<std::string, std::vector<float>> key_commands = {
        {"w", {1.0f, 0.0f, 0.0f}},
        {"s", {-1.0f, 0.0f, 0.0f}},
        {"a", {0.0f, 1.0f, 0.0f}},
        {"d", {0.0f, -1.0f, 0.0f}},
        {"q", {0.0f, 0.0f, 1.0f}},
        {"e", {0.0f, 0.0f, -1.0f}}
    };
    std::vector<float> cmd = {0.0f, 0.0f, 0.0f};
    if (key_commands.find(key) != key_commands.end())
    {
        cmd = key_commands[key];
        cmd[0] = std::clamp(cmd[0], cfg["lin_vel_x"][0].as<float>(), cfg["lin_vel_x"][1].as<float>());
        cmd[1] = std::clamp(cmd[1], cfg["lin_vel_y"][0].as<float>(), cfg["lin_vel_y"][1].as<float>());
        cmd[2] = std::clamp(cmd[2], cfg["ang_vel_z"][0].as<float>(), cfg["ang_vel_z"][1].as<float>());
    }
    return cmd;
}

}

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    auto deploy_cfg = YAML::LoadFile(policy_dir / "params" / "deploy.yaml");
    if (cfg["keyboard_control"].as<bool>(false))
    {
        // Rebuild in the same order to preserve the policy's observation layout.
        YAML::Node observations(YAML::NodeType::Map);
        for (const auto& term : deploy_cfg["observations"])
        {
            auto name = term.first.as<std::string>();
            if (name == "velocity_commands") name = "keyboard_velocity_commands";
            observations[name] = term.second;
        }
        deploy_cfg["observations"] = observations;
    }

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        deploy_cfg,
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}
