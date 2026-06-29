#include <memory>
#include <vector>
#include <string>
#include <map>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"

#include "control_msgs/action/follow_joint_trajectory.hpp"
#include "trajectory_msgs/msg/joint_trajectory_point.hpp"
#include "builtin_interfaces/msg/duration.hpp"

using FollowJointTrajectory = control_msgs::action::FollowJointTrajectory;

class TaskPlanner : public rclcpp::Node
{
public:
    TaskPlanner()
        : Node("task_planner"),
          current_state_(0)
    {
        arm_client_ = rclcpp_action::create_client<FollowJointTrajectory>(
            this,
            "/arm_controller/follow_joint_trajectory");

        gripper_client_ = rclcpp_action::create_client<FollowJointTrajectory>(
            this,
            "/gripper_controller/follow_joint_trajectory");

        arm_positions_["home"]  = {0.0,  0.0,  0.0};
        arm_positions_["pick"]  = {0.8,  0.5, -0.5};
        arm_positions_["place"] = {-0.8, 0.5, -0.5};

        GRIPPER_OPEN_ = 0.0;
        GRIPPER_CLOSED_ = -1.2;

        states_ = {
            {"arm",     "home",   2},
            {"gripper", "open",   1},
            {"arm",     "pick",   2},
            {"gripper", "close",  1},
            {"arm",     "place",  2},
            {"gripper", "open",   1},
            {"arm",     "home",   2}
        };

        RCLCPP_INFO(this->get_logger(),
                    "Task Planner started — Waiting for action servers...");

        arm_client_->wait_for_action_server();
        gripper_client_->wait_for_action_server();

        RCLCPP_INFO(this->get_logger(),
                    "Action servers ready — starting sequence");

        executeNextState();
    }

private:

    // ==========================================================
    struct State
    {
        std::string controller;
        std::string action;
        int duration;
    };

    // ==========================================================
    void executeNextState()
    {
        if (current_state_ >= states_.size())
        {
            RCLCPP_INFO(this->get_logger(), "Sequence completed");
            return;
        }

        const auto &state = states_[current_state_];

        RCLCPP_INFO(this->get_logger(),
                    "State %zu/%zu: %s → %s",
                    current_state_ + 1,
                    states_.size(),
                    state.controller.c_str(),
                    state.action.c_str());

        if (state.controller == "arm")
        {
            sendArmGoal(
                arm_positions_[state.action],
                state.duration);
        }
        else if (state.controller == "gripper")
        {
            double position =
                (state.action == "open")
                    ? GRIPPER_OPEN_
                    : GRIPPER_CLOSED_;

            sendGripperGoal(position, state.duration);
        }
    }

    // ==========================================================
    void sendArmGoal(const std::vector<double> &positions, int duration_sec)
    {
        FollowJointTrajectory::Goal goal;
        goal.trajectory.joint_names = {"joint_1", "joint_2", "joint_3"};

        trajectory_msgs::msg::JointTrajectoryPoint point;
        point.positions = positions;

        builtin_interfaces::msg::Duration dur;
        dur.sec = duration_sec;
        point.time_from_start = dur;

        goal.trajectory.points.push_back(point);

        RCLCPP_INFO(this->get_logger(), "Arm → goal sent");

        auto options =
            rclcpp_action::Client<FollowJointTrajectory>::SendGoalOptions();

        options.goal_response_callback =
            std::bind(&TaskPlanner::goalResponseCallback, this, std::placeholders::_1);

        options.result_callback =
            std::bind(&TaskPlanner::resultCallback, this, std::placeholders::_1);

        arm_client_->async_send_goal(goal, options);
    }

    // ==========================================================
    void sendGripperGoal(double position, int duration_sec)
    {
        FollowJointTrajectory::Goal goal;
        goal.trajectory.joint_names = {"joint_4"};

        trajectory_msgs::msg::JointTrajectoryPoint point;
        point.positions = {position};

        builtin_interfaces::msg::Duration dur;
        dur.sec = duration_sec;
        point.time_from_start = dur;

        goal.trajectory.points.push_back(point);

        std::string state = (position >= 0.0) ? "OPEN" : "CLOSED";

        RCLCPP_INFO(this->get_logger(),
                    "Gripper → %s (%.2f rad)",
                    state.c_str(),
                    position);

        auto options =
            rclcpp_action::Client<FollowJointTrajectory>::SendGoalOptions();

        options.goal_response_callback =
            std::bind(&TaskPlanner::goalResponseCallback, this, std::placeholders::_1);

        options.result_callback =
            std::bind(&TaskPlanner::resultCallback, this, std::placeholders::_1);

        gripper_client_->async_send_goal(goal, options);
    }

    // ==========================================================
    void goalResponseCallback(
    rclcpp_action::ClientGoalHandle<FollowJointTrajectory>::SharedPtr goal_handle)
    {
        if (!goal_handle)
            {
                RCLCPP_ERROR(this->get_logger(), "Goal refused");
                return;
            }
        RCLCPP_INFO(this->get_logger(), "Goal accepted");
    }

    // ==========================================================
    void resultCallback(
        const rclcpp_action::ClientGoalHandle<FollowJointTrajectory>::WrappedResult &result)
    {
        if (result.code == rclcpp_action::ResultCode::SUCCEEDED)
        {
            RCLCPP_INFO(this->get_logger(), "Goal completed");
        }
        else
        {
            RCLCPP_WARN(this->get_logger(),
                        "Goal failed or aborted (code: %d)",
                        static_cast<int>(result.code));
        }

        current_state_++;
        executeNextState();
    }

private:

    // Clients
    rclcpp_action::Client<FollowJointTrajectory>::SharedPtr arm_client_;
    rclcpp_action::Client<FollowJointTrajectory>::SharedPtr gripper_client_;

    // State machine
    std::vector<State> states_;
    size_t current_state_;

    // Parameters
    std::map<std::string, std::vector<double>> arm_positions_;

    double GRIPPER_OPEN_;
    double GRIPPER_CLOSED_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<TaskPlanner>();

    rclcpp::spin(node);

    rclcpp::shutdown();
    return 0;
}