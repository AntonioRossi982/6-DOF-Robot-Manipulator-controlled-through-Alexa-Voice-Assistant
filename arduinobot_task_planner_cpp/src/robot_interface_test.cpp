#include <memory>
#include <vector>
#include <string>
#include <thread>
#include <atomic>
#include <chrono>
#include <iostream>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"

#include "control_msgs/action/follow_joint_trajectory.hpp"
#include "trajectory_msgs/msg/joint_trajectory_point.hpp"
#include "builtin_interfaces/msg/duration.hpp"

using FollowJointTrajectory = control_msgs::action::FollowJointTrajectory;

class RobotInterface : public rclcpp::Node
{
public:
    RobotInterface()
        : Node("robot_interface"),
          is_executing_(false),
          pending_duration_(1.0)
    {
        // Action clients
        arm_client_ = rclcpp_action::create_client<FollowJointTrajectory>(
            this,
            "/arm_controller/follow_joint_trajectory");

        gripper_client_ = rclcpp_action::create_client<FollowJointTrajectory>(
            this,
            "/gripper_controller/follow_joint_trajectory");

        // Presets
        arm_presets_["home"]  = {0.0,  0.0,  0.0};
        arm_presets_["pick"]  = {0.8,  0.5, -0.5};
        arm_presets_["place"] = {-0.8, 0.5, -0.5};

        GRIPPER_OPEN_ = 0.0;
        GRIPPER_CLOSED_ = -1.2;

        RCLCPP_INFO(this->get_logger(), "Waiting for action servers...");

        if (!arm_client_->wait_for_action_server(std::chrono::seconds(5)) ||
            !gripper_client_->wait_for_action_server(std::chrono::seconds(5)))
        {
            RCLCPP_WARN(this->get_logger(), "Action servers not ready yet");
        }

        RCLCPP_INFO(this->get_logger(), "Action servers ready");

        printHelp();

        input_thread_ = std::thread(&RobotInterface::inputLoop, this);
        input_thread_.detach();
    }

private:

    // =========================
    // ACTION SEND ARM
    // =========================
    void sendArm(const std::vector<double> &positions, double duration_sec = 2.0)
    {
        if (is_executing_)
        {
            std::cout << "Robot moving — wait...\n";
            return;
        }

        FollowJointTrajectory::Goal goal;
        goal.trajectory.joint_names = {"joint_1", "joint_2", "joint_3"};

        trajectory_msgs::msg::JointTrajectoryPoint point;
        point.positions = positions;

        builtin_interfaces::msg::Duration dur;
        dur.sec = static_cast<int32_t>(duration_sec);
        point.time_from_start = dur;

        goal.trajectory.points.push_back(point);

        pending_duration_ = duration_sec;
        is_executing_ = true;

        RCLCPP_INFO(this->get_logger(), "Arm command sent");

        auto send_goal_options =
            rclcpp_action::Client<FollowJointTrajectory>::SendGoalOptions();

        send_goal_options.goal_response_callback =
            std::bind(&RobotInterface::goalResponseCallback, this, std::placeholders::_1);

        send_goal_options.result_callback =
            std::bind(&RobotInterface::resultCallback, this, std::placeholders::_1);

        arm_client_->async_send_goal(goal, send_goal_options);
    }

    // =========================
    // ACTION SEND GRIPPER
    // =========================
    void sendGripper(double position, double duration_sec = 2.0)
    {
        if (is_executing_)
        {
            std::cout << "Robot moving — wait...\n";
            return;
        }

        FollowJointTrajectory::Goal goal;
        goal.trajectory.joint_names = {"joint_4"};

        trajectory_msgs::msg::JointTrajectoryPoint point;
        point.positions = {position};

        builtin_interfaces::msg::Duration dur;
        dur.sec = static_cast<int32_t>(duration_sec);
        point.time_from_start = dur;

        goal.trajectory.points.push_back(point);

        pending_duration_ = duration_sec;
        is_executing_ = true;

        std::string state = (position >= 0.0) ? "OPEN" : "CLOSED";
        RCLCPP_INFO(this->get_logger(), "Gripper → %s", state.c_str());

        auto send_goal_options =
            rclcpp_action::Client<FollowJointTrajectory>::SendGoalOptions();

        send_goal_options.goal_response_callback =
            std::bind(&RobotInterface::goalResponseCallback, this, std::placeholders::_1);

        send_goal_options.result_callback =
            std::bind(&RobotInterface::resultCallback, this, std::placeholders::_1);

        gripper_client_->async_send_goal(goal, send_goal_options);
    }

    // =========================
    // CALLBACKS
    // =========================
    void goalResponseCallback(
        rclcpp_action::ClientGoalHandle<FollowJointTrajectory>::SharedPtr goal_handle)
    {
        if (!goal_handle)
        {
            RCLCPP_ERROR(this->get_logger(), "Goal rejected");
            is_executing_ = false;
            return;
        }

        RCLCPP_INFO(this->get_logger(), "Goal accepted");
    }

    void resultCallback(
        const rclcpp_action::ClientGoalHandle<FollowJointTrajectory>::WrappedResult &)
    {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(static_cast<int>(pending_duration_ * 1000)));

        is_executing_ = false;
        RCLCPP_INFO(this->get_logger(), "Movement completed");
    }

    // =========================
    // TASK SEQUENCE
    // =========================
    void executeTask()
    {
        RCLCPP_INFO(this->get_logger(), "Starting pick and place sequence...");

        struct Step
        {
            std::string type;
            std::vector<double> arm_target;
            double gripper_target;
            bool is_arm;
            double duration;
        };

        std::vector<Step> steps = {
            {"arm",     arm_presets_["pick"],  0.0, true,  2},
            {"gripper", {}, GRIPPER_OPEN_,     false, 2},
            {"gripper", {}, GRIPPER_CLOSED_,   false, 2},
            {"arm",     arm_presets_["place"], 0.0, true,  2},
            {"gripper", {}, GRIPPER_OPEN_,     false, 2},
            {"gripper", {}, GRIPPER_CLOSED_,   false, 2},
            {"arm",     arm_presets_["home"],  0.0, true,  2},
        };

        for (auto &s : steps)
        {
            while (is_executing_)
                std::this_thread::sleep_for(std::chrono::milliseconds(100));

            if (s.is_arm)
                sendArm(s.arm_target, s.duration);
            else
                sendGripper(s.gripper_target, s.duration);

            std::this_thread::sleep_for(
                std::chrono::milliseconds(static_cast<int>(s.duration * 1000 + 500)));
        }

        while (is_executing_)
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

        RCLCPP_INFO(this->get_logger(), "Pick and place sequence completed");
    }

    // =========================
    // INPUT LOOP
    // =========================
    void inputLoop()
    {
        std::string cmd;

        while (rclcpp::ok())
        {
            std::cout << "\n> ";
            std::getline(std::cin, cmd);

            for (auto &c : cmd)
                c = std::tolower(c);

            if (arm_presets_.count(cmd))
            {
                sendArm(arm_presets_[cmd]);
            }
            else if (cmd == "open")
            {
                sendGripper(GRIPPER_OPEN_);
            }
            else if (cmd == "close")
            {
                sendGripper(GRIPPER_CLOSED_);
            }
            else if (cmd == "task")
            {
                if (is_executing_)
                {
                    std::cout << "Robot moving — wait...\n";
                }
                else
                {
                    std::thread(&RobotInterface::executeTask, this).detach();
                }
            }
            else if (cmd == "help")
            {
                printHelp();
            }
            else if (cmd == "quit")
            {
                RCLCPP_INFO(this->get_logger(), "Exiting program");
                rclcpp::shutdown();
                break;
            }
            else
            {
                std::cout << "Unknown command: " << cmd << "\n";
                std::cout << "Type 'help' for available commands\n";
            }
        }
    }

    // =========================
    // HELP
    // =========================
    void printHelp()
    {
        std::cout << "\n============================\n";
        std::cout << "   Arduinobot Interface\n";
        std::cout << "============================\n";
        std::cout << " home   — home position\n";
        std::cout << " pick   — pick position\n";
        std::cout << " place  — place position\n";
        std::cout << " open   — open gripper\n";
        std::cout << " close  — close gripper\n";
        std::cout << " task   — pick and place sequence\n";
        std::cout << " help   — show menu\n";
        std::cout << " quit   — exit\n";
        std::cout << "============================\n";
    }

private:

    // Action clients
    rclcpp_action::Client<FollowJointTrajectory>::SharedPtr arm_client_;
    rclcpp_action::Client<FollowJointTrajectory>::SharedPtr gripper_client_;

    // State
    std::atomic<bool> is_executing_;
    double pending_duration_;

    // Thread
    std::thread input_thread_;

    // Presets
    std::map<std::string, std::vector<double>> arm_presets_;

    double GRIPPER_OPEN_;
    double GRIPPER_CLOSED_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<RobotInterface>();

    rclcpp::spin(node);

    rclcpp::shutdown();
    return 0;
}