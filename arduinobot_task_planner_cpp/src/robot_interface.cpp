#include <memory>
#include <vector>
#include <string>
#include <thread>
#include <atomic>
#include <chrono>
#include <iostream>
#include <map>

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
          task_running_(false)
    {
        arm_client_ = rclcpp_action::create_client<FollowJointTrajectory>(
            this,
            "/arm_controller/follow_joint_trajectory");

        gripper_client_ = rclcpp_action::create_client<FollowJointTrajectory>(
            this,
            "/gripper_controller/follow_joint_trajectory");

        arm_presets_["home"]  = {0.0,  0.0,  0.0};
        arm_presets_["pick"]  = {0.8,  0.5, -0.5};
        arm_presets_["place"] = {-0.8, 0.5, -0.5};

        GRIPPER_CLOSED_ = 0.0;
        GRIPPER_OPEN_   = -1.45;

        RCLCPP_INFO(this->get_logger(), "Waiting for action servers...");

        arm_client_->wait_for_action_server();
        gripper_client_->wait_for_action_server();

        RCLCPP_INFO(this->get_logger(), "Action servers ready");

        printHelp();

        input_thread_ = std::thread(&RobotInterface::inputLoop, this);
        input_thread_.detach();
    }

private:

    // ==========================================================
    // GENERIC ACTION EXECUTION (SYNC STYLE LIKE PYTHON)
    // ==========================================================
    bool executeActionAndWait(
        rclcpp_action::Client<FollowJointTrajectory>::SharedPtr client,
        const std::vector<std::string> &joint_names,
        const std::vector<double> &positions,
        double duration_sec)
    {
        FollowJointTrajectory::Goal goal;
        goal.trajectory.joint_names = joint_names;

        trajectory_msgs::msg::JointTrajectoryPoint point;
        point.positions = positions;

        builtin_interfaces::msg::Duration dur;
        dur.sec = static_cast<int32_t>(duration_sec);
        point.time_from_start = dur;

        goal.trajectory.points.push_back(point);

        auto send_future = client->async_send_goal(goal);

        while (rclcpp::ok() && !send_future.valid())
            std::this_thread::sleep_for(std::chrono::milliseconds(10));

        auto goal_handle = send_future.get();

        if (!goal_handle)
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to communicate with action server");
            return false;
        }

        if (!goal_handle->accept_result())
        // if (!goal_handle)
        {
            RCLCPP_ERROR(this->get_logger(), "Goal rejected");
            return false;
        }

        RCLCPP_INFO(this->get_logger(), "Goal accepted");

        auto result_future = goal_handle->async_result();

        while (rclcpp::ok() && !result_future.valid())
            std::this_thread::sleep_for(std::chrono::milliseconds(10));

        auto result = result_future.get();

        if (!result)
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to receive action result");
            return false;
        }

        RCLCPP_INFO(this->get_logger(), "Movement completed");
        return true;
    }

    // ==========================================================
    // ARM
    // ==========================================================
    bool moveArm(const std::vector<double> &positions, double duration_sec = 2.0)
    {
        RCLCPP_INFO(this->get_logger(), "Arm -> ...");

        return executeActionAndWait(
            arm_client_,
            {"joint_1", "joint_2", "joint_3"},
            positions,
            duration_sec);
    }

    // ==========================================================
    // GRIPPER
    // ==========================================================
    bool moveGripper(double position, double duration_sec = 2.0)
    {
        std::string state = (position == GRIPPER_OPEN_) ? "OPEN" : "CLOSED";

        RCLCPP_INFO(this->get_logger(), "Gripper -> %s", state.c_str());

        return executeActionAndWait(
            gripper_client_,
            {"joint_4"},
            {position},
            duration_sec);
    }

    // ==========================================================
    // TASK SEQUENCE
    // ==========================================================
    void executeTask()
    {
        if (task_running_)
        {
            RCLCPP_WARN(this->get_logger(), "Task already running");
            return;
        }

        task_running_ = true;

        RCLCPP_INFO(this->get_logger(), "Starting pick and place sequence...");

        try
        {
            if (!moveArm(arm_presets_["pick"])) return;
            if (!moveGripper(GRIPPER_OPEN_)) return;
            if (!moveGripper(GRIPPER_CLOSED_)) return;

            if (!moveArm(arm_presets_["place"])) return;
            if (!moveGripper(GRIPPER_OPEN_)) return;
            if (!moveGripper(GRIPPER_CLOSED_)) return;

            if (!moveArm(arm_presets_["home"])) return;

            RCLCPP_INFO(this->get_logger(), "Pick and place sequence completed");
        }
        catch (...)
        {
            RCLCPP_ERROR(this->get_logger(), "Unexpected error in task");
        }

        task_running_ = false;
    }

    // ==========================================================
    // INPUT LOOP
    // ==========================================================
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
                std::thread(&RobotInterface::moveArm, this, arm_presets_[cmd])
                    .detach();
            }
            else if (cmd == "open")
            {
                std::thread(&RobotInterface::moveGripper, this, GRIPPER_OPEN_)
                    .detach();
            }
            else if (cmd == "close")
            {
                std::thread(&RobotInterface::moveGripper, this, GRIPPER_CLOSED_)
                    .detach();
            }
            else if (cmd == "task")
            {
                std::thread(&RobotInterface::executeTask, this).detach();
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

    // ==========================================================
    // HELP
    // ==========================================================
    void printHelp()
    {
        std::cout << "\n============================\n";
        std::cout << "   Arduinobot Interface\n";
        std::cout << "============================\n";
        std::cout << " home  - home position\n";
        std::cout << " pick  - pick position\n";
        std::cout << " place - place position\n";
        std::cout << " open  - open gripper\n";
        std::cout << " close - close gripper\n";
        std::cout << " task  - pick and place\n";
        std::cout << " help  - show menu\n";
        std::cout << " quit  - exit\n";
        std::cout << "============================\n";
    }

private:

    // Clients
    rclcpp_action::Client<FollowJointTrajectory>::SharedPtr arm_client_;
    rclcpp_action::Client<FollowJointTrajectory>::SharedPtr gripper_client_;

    // State
    std::atomic<bool> task_running_;

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