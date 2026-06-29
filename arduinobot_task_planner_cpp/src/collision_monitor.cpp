#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/bool.hpp"

class CollisionMonitor : public rclcpp::Node
{
public:
    CollisionMonitor()
        : Node("collision_monitor")
    {
        // Subscriber
        joint_state_subscriber_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states",
            10,
            std::bind(&CollisionMonitor::jointStateCallback, this, std::placeholders::_1));

        // Publisher
        danger_publisher_ = this->create_publisher<std_msgs::msg::Bool>(
            "/collision_warning",
            10);

        // Obstacle position
        obstacle_x_ = 0.15;
        obstacle_y_ = 0.15;
        obstacle_z_ = 0.15;

        // Distance threshold
        DANGER_THRESHOLD_ = 0.2;

        // Robot link lengths
        L1_ = 0.12;
        L2_ = 0.12;
        L3_ = 0.08;

        RCLCPP_INFO(
            this->get_logger(),
            "Collision monitor started - threshold: %.2f m",
            DANGER_THRESHOLD_);
    }

private:

    void forwardKinematics(
        double joint1,
        double joint2,
        double joint3,
        double &x,
        double &y,
        double &z)
    {
        x = L1_ * std::cos(joint1)
          + L2_ * std::cos(joint1 + joint2)
          + L3_ * std::cos(joint1 + joint2 + joint3);

        y = L1_ * std::sin(joint1)
          + L2_ * std::sin(joint1 + joint2)
          + L3_ * std::sin(joint1 + joint2 + joint3);

        z = 0.0;
    }

    double distanceToObstacle(double x, double y, double z)
    {
        double dx = x - obstacle_x_;
        double dy = y - obstacle_y_;
        double dz = z - obstacle_z_;

        return std::sqrt(dx * dx + dy * dy + dz * dz);
    }

    void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        auto findJointIndex = [&](const std::string &name) -> int
        {
            auto it = std::find(msg->name.begin(), msg->name.end(), name);

            if (it == msg->name.end())
                return -1;

            return std::distance(msg->name.begin(), it);
        };

        int joint1_index = findJointIndex("joint_1");
        int joint2_index = findJointIndex("joint_2");
        int joint3_index = findJointIndex("joint_3");

        if (joint1_index < 0 || joint2_index < 0 || joint3_index < 0)
            return;

        double joint1 = msg->position[joint1_index];
        double joint2 = msg->position[joint2_index];
        double joint3 = msg->position[joint3_index];

        double x, y, z;
        forwardKinematics(joint1, joint2, joint3, x, y, z);

        double distance = distanceToObstacle(x, y, z);

        std_msgs::msg::Bool warning;
        warning.data = (distance < DANGER_THRESHOLD_);

        danger_publisher_->publish(warning);

        if (warning.data)
        {
            RCLCPP_WARN(
                this->get_logger(),
                "WARNING - obstacle distance: %.3f m (threshold: %.3f m)",
                distance,
                DANGER_THRESHOLD_);
        }
        else
        {
            RCLCPP_INFO_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "Obstacle distance: %.3f m - SAFE DISTANCE",
                distance);
        }
    }

    // Subscribers
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_subscriber_;

    // Publishers
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr danger_publisher_;

    // Obstacle position
    double obstacle_x_;
    double obstacle_y_;
    double obstacle_z_;

    // Robot parameters
    double L1_;
    double L2_;
    double L3_;

    // Threshold
    double DANGER_THRESHOLD_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<CollisionMonitor>();

    rclcpp::spin(node);

    rclcpp::shutdown();

    return 0;
}