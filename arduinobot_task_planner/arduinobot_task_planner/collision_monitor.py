import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
import math


class CollisionMonitor(Node):
    def __init__(self):
        super().__init__('collision_monitor')

        # Subscriber to joint states
        self.joint_state_subscriber = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # Publisher for danger collision warnings
        self.danger_publisher = self.create_publisher(
            Bool,
            '/collision_warning',
            10
        )

        # Obstacle position - Must match the one defined in scene_manager.py
        self.obstacle_position = {
            'x': 0.15,
            'y': 0.15,
            'z': 0.15
        }

        # Distance threshold in meters - A warning is triggered below this distance
        self.DANGER_THRESHOLD = 0.2  # previous value was 0.15

        # Robot link lengths in meters - Adjust these values to match your physical robot
        self.L1 = 0.12  # link_1 length
        self.L2 = 0.12  # link_2 length
        self.L3 = 0.08  # link_3 length

        self.get_logger().info(
            f'The monitoring collision node started — threshold: {self.DANGER_THRESHOLD}m'
        )

    def forward_kinematics(self, joint1, joint2, joint3):
        """
        Simplified forward kinematics used to estimate
        the end-effector position (in x-y plane).
        """
        x = (self.L1 * math.cos(joint1) +
             self.L2 * math.cos(joint1 + joint2) +
             self.L3 * math.cos(joint1 + joint2 + joint3))

        y = (self.L1 * math.sin(joint1) +
             self.L2 * math.sin(joint1 + joint2) +
             self.L3 * math.sin(joint1 + joint2 + joint3))

        z = 0.0  # planar simplification

        return x, y, z

    def distance_to_obstacle(self, x, y, z):
        """Compute the Euclidean distance from the obstacle."""
        dx = x - self.obstacle_position['x']
        dy = y - self.obstacle_position['y']
        dz = z - self.obstacle_position['z']

        return math.sqrt(dx**2 + dy**2 + dz**2)

    def joint_state_callback(self, msg):
        # Extract joint positions
        try:
            joint1_index = msg.name.index('joint_1')
            joint2_index = msg.name.index('joint_2')
            joint3_index = msg.name.index('joint_3')
        except ValueError:
            return

        joint1 = msg.position[joint1_index]
        joint2 = msg.position[joint2_index]
        joint3 = msg.position[joint3_index]

        # Compute end-effector position
        x, y, z = self.forward_kinematics(joint1, joint2, joint3)

        # Compute distance to the obstacle
        distance = self.distance_to_obstacle(x, y, z)

        # Publish warning if robot distance to obstacle is below the threshold
        warning = Bool()
        warning.data = distance < self.DANGER_THRESHOLD
        self.danger_publisher.publish(warning)

        if warning.data:
            self.get_logger().warn(
                f'WARNING — obstacle distance: {distance:.3f}m '
                f'(threshold: {self.DANGER_THRESHOLD}m)'
            )
        else:
            self.get_logger().info(
                f'Obstacle distance: {distance:.3f}m — SAFE DISTANCE',
                throttle_duration_sec=2.0
            )


def main(args=None):
    rclpy.init(args=args)
    node = CollisionMonitor()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()