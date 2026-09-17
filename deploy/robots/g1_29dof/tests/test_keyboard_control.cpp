#include <pty.h>
#include <cmath>
#include "../src/State_RLBase.cpp"

std::unique_ptr<LowCmd_t> FSMState::lowcmd;
std::shared_ptr<LowState_t> FSMState::lowstate;
std::shared_ptr<Keyboard> FSMState::keyboard;

void require(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

int main()
{
    int master, slave;
    require(openpty(&master, &slave, nullptr, nullptr, nullptr) == 0, "openpty failed");
    int saved_stdin = dup(STDIN_FILENO);
    dup2(slave, STDIN_FILENO);
    FSMState::keyboard = std::make_shared<Keyboard>();

    auto press = [&](char key) {
        require(write(master, &key, 1) == 1, "PTY write failed");
        for (int i = 0; i < 100 && FSMState::keyboard->key() != std::string(1, key); ++i)
            usleep(1000);
        FSMState::keyboard->update();
        require(FSMState::keyboard->key() == std::string(1, key), "key not received");
    };

    try {
        param::config = YAML::LoadFile("config/config.yaml");
        for (const auto& entry : param::config["FSM"]["_"])
            FSMStringMap.insert({entry.second["id"].as<int>(), entry.first.as<std::string>()});

        // Isolate keyboard transitions from DDS joystick and timeout checks.
        for (const auto& name : {"Passive", "FixStand", "Velocity"})
            param::config["FSM"][name].remove("transitions");
        FSMState passive(1, "Passive"), stand(2, "FixStand"), velocity(3, "Velocity");
        auto target = [](FSMState& state) {
            for (size_t i = 0; i + 1 < state.registered_checks.size(); ++i)
                if (state.registered_checks[i].first()) return state.registered_checks[i].second;
            return 0;
        };
        press('2');
        require(target(passive) == 0, "Passive must not jump directly to Velocity");
        press('1');
        require(target(passive) == 2, "1 must enter FixStand from Passive");
        FSMState::keyboard->update();
        require(target(passive) == 0, "held key must not retrigger a state transition");
        press('2');
        require(target(stand) == 3, "2 must enter Velocity from FixStand");
        press('0');
        require(target(stand) == 1 && target(velocity) == 1, "0 must return to Passive");
        press('w');
        require(target(velocity) == 0, "movement key must not change state");

        auto cfg = YAML::LoadFile("config/policy/velocity/windows_trained/params/deploy.yaml");
        // Exercise the actual registered keyboard observation without robot/DDS hardware.
        auto command_cfg = YAML::Clone(cfg["observations"]["velocity_commands"]);
        cfg["observations"] = YAML::Node(YAML::NodeType::Map);
        cfg["observations"]["keyboard_velocity_commands"] = command_cfg;
        auto robot = std::make_shared<isaaclab::Articulation>();
        isaaclab::ManagerBasedRLEnv env(cfg, robot);
        const std::vector<std::pair<char, std::vector<float>>> cases = {
            {'w', {1.0f, 0, 0}}, {'s', {-0.5f, 0, 0}},
            {'a', {0, 0.3f, 0}}, {'d', {0, -0.3f, 0}},
            {'q', {0, 0, 0.2f}}, {'e', {0, 0, -0.2f}},
            {'2', {0, 0, 0}}, {'x', {0, 0, 0}}
        };
        for (const auto& [key, expected] : cases) {
            press(key);
            const auto actual = isaaclab::keyboard_velocity_commands(&env, YAML::Node());
            for (size_t i = 0; i < 3; ++i)
                require(std::abs(actual[i] - expected[i]) < 1e-6f,
                        std::string("wrong velocity or missing range limit for ") + key);
        }
        press('w');
        usleep(120000);
        require(isaaclab::keyboard_velocity_commands(&env, YAML::Node()) == std::vector<float>(3, 0),
                "input timeout must zero velocity");
        env.cfg["commands"]["base_velocity"]["ranges"]["lin_vel_x"][1] = 0.15f;
        press('w');
        require(std::abs(isaaclab::keyboard_velocity_commands(&env, YAML::Node())[0] - 0.15f) < 1e-6f,
                "keyboard speed must honor the active policy's ranges");
        for (const auto& sequence : {std::string("\033"), std::string("\033[")}) {
            press('w');
            require(write(master, sequence.data(), sequence.size()) == static_cast<ssize_t>(sequence.size()),
                    "PTY Escape write failed");
            usleep(200000);
            require(isaaclab::keyboard_velocity_commands(&env, YAML::Node()) == std::vector<float>(3, 0),
                    "incomplete Escape sequence must not leave a movement command stuck");
        }
        FSMState::keyboard.reset();
        dup2(saved_stdin, STDIN_FILENO);
        close(saved_stdin);
        close(master);
        close(slave);
        std::cout << "Keyboard transitions, velocity ranges and timeout passed.\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        std::_Exit(1);
    }
}
