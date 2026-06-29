#include <memory>
#include <vector>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "moveit_msgs/msg/collision_object.hpp"
#include "shape_msgs/msg/solid_primitive.hpp"
#include "geometry_msgs/msg/pose.hpp"

using moveit_msgs::msg::CollisionObject;

class SceneManager : public rclcpp::Node
{
public:
    SceneManager()
        : Node("scene_manager"),
          publish_count_(0),
          max_publishes_(5)
    {
        collision_pub_ = this->create_publisher<CollisionObject>(
            "/collision_object", 10);

        RCLCPP_INFO(this->get_logger(),
                    "Waiting for MoveIt2 — 10 seconds to start...");

        start_timer_ = this->create_wall_timer(
            std::chrono::seconds(10),
            std::bind(&SceneManager::startPublishing, this));
    }

private:

    // ==========================================================
    // START PERIODIC PUBLISHING
    // ==========================================================
    void startPublishing()
    {
        start_timer_->cancel();

        publish_timer_ = this->create_wall_timer(
            std::chrono::seconds(2),
            std::bind(&SceneManager::publishObjects, this));
    }

    // ==========================================================
    // MAIN LOOP
    // ==========================================================
    void publishObjects()
    {
        if (publish_count_ >= max_publishes_)
        {
            publish_timer_->cancel();
            RCLCPP_INFO(this->get_logger(), "Publishing completed");
            return;
        }

        publish_count_++;

        RCLCPP_INFO(this->get_logger(),
                    "Publishing %d/%d",
                    publish_count_,
                    max_publishes_);

        addTable();
        addBox();
    }

    // ==========================================================
    // TABLE OBJECT
    // ==========================================================
    void addTable()
    {
        CollisionObject obj;
        obj.id = "table";
        obj.header.frame_id = "world";
        obj.header.stamp = this->now();

        shape_msgs::msg::SolidPrimitive shape;
        shape.type = shape_msgs::msg::SolidPrimitive::BOX;
        shape.dimensions = {0.6, 0.6, 0.05};

        geometry_msgs::msg::Pose pose;
        pose.position.x = 0.0;
        pose.position.y = 0.0;
        pose.position.z = -0.05;
        pose.orientation.w = 1.0;

        obj.primitives.push_back(shape);
        obj.primitive_poses.push_back(pose);
        obj.operation = CollisionObject::ADD;

        collision_pub_->publish(obj);
    }

    // ==========================================================
    // OBSTACLE BOX
    // ==========================================================
    void addBox()
    {
        CollisionObject obj;
        obj.id = "obstacle";
        obj.header.frame_id = "world";
        obj.header.stamp = this->now();

        shape_msgs::msg::SolidPrimitive shape;
        shape.type = shape_msgs::msg::SolidPrimitive::BOX;
        shape.dimensions = {0.05, 0.05, 0.3};

        geometry_msgs::msg::Pose pose;
        pose.position.x = 0.15;
        pose.position.y = 0.15;
        pose.position.z = 0.15;
        pose.orientation.w = 1.0;

        obj.primitives.push_back(shape);
        obj.primitive_poses.push_back(pose);
        obj.operation = CollisionObject::ADD;

        collision_pub_->publish(obj);
    }

private:

    // Publisher
    rclcpp::Publisher<CollisionObject>::SharedPtr collision_pub_;

    // Timers
    rclcpp::TimerBase::SharedPtr start_timer_;
    rclcpp::TimerBase::SharedPtr publish_timer_;

    // State
    int publish_count_;
    int max_publishes_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<SceneManager>();

    rclcpp::spin(node);

    rclcpp::shutdown();
    return 0;
}