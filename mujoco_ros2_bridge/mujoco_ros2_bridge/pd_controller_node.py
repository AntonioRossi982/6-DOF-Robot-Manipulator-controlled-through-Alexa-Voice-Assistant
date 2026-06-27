import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
import numpy as np
import sys
import os

# Import the PDController from the MuJoCo project
sys.path.insert(0, os.path.expanduser('~/mujoco_robotics'))
from controllers.pd_controller import PDController

class PDControllerNode(Node):
    """
    ROS2 node that implements the PD controller.
    
    It reads joint states from /mujoco/joint_states,
    calculates the PD control, and publishes commands
    to /mujoco/joint_commands.
    
    The target can be updated via /target_position.
    
    Important note: this node does NOT know that it is controlling
    MuJoCo — it could control a real robot using the
    exact same ROS2 interface.
    """

    def __init__(self):
        super().__init__('pd_controller')

        # ROS2 parameters — configurable without recompiling
        # Declaration with default values
        self.declare_parameter('kp', 150.0)
        self.declare_parameter('kd', 15.0)
        self.declare_parameter('n_joints', 4)
        self.declare_parameter('tolerance', 0.02)

        # Read the parameters
        kp = self.get_parameter('kp').value
        kd = self.get_parameter('kd').value
        n_joints = self.get_parameter('n_joints').value
        self.tolerance = self.get_parameter('tolerance').value

        # Initialize the PDController
        self.controller = PDController(kp=kp, kd=kd, n_joints=n_joints)

        # Initial target - home position
        self.controller.set_target(np.zeros(n_joints))
        self.get_logger().info(f'Controller PD: Kp={kp}, Kd={kd}, ' f'n_joints={n_joints}')

        # Subscriber for joint states
        self.joint_state_sub = self.create_subscription(JointState, '/mujoco/joint_states', self.joint_state_callback, 10)

        # Subscriber for the position target
        # Format: "j1_rad,j2_rad" e.g., "0.5,0.3"
        self.target_sub = self.create_subscription(String, '/target_position', self.target_callback, 10)

        # Publisher for the control commands
        self.cmd_pub = self.create_publisher(Float64MultiArray, '/mujoco/joint_commands', 10)

        # Publisher for the controller status
        self.status_pub = self.create_publisher(String, '/controller_status', 10)
        self.get_logger().info('Controller PD ready')

    def target_callback(self, msg):
        """
        Callback for updating the position target.
        Message format: "joint1, joint2, joint3, ..." in radians
        Each value is the target position of a joint in radians.
        """
        try:
            values = [float(v) for v in msg.data.split(',')]
            self.controller.set_target(values)
            self.get_logger().info(f'New target: {values} rad')
        except Exception as e:
            self.get_logger().error(f'Target not valid: {msg.data} — {e}')

    def joint_state_callback(self, msg):
        """
        Main callback — called at every /joint_states message.
        Sequence:
        1. Extract positions and velocities from the message
        2. Calculate the PD control
        3. Publish the command
        4. Publish the controller status
        """
        # Extract positions and velocities
        q = np.array(msg.position)
        qd = np.array(msg.velocity)

        # Check if the dimensions match the number of joints (no underactuated joints)
        n = self.controller.n_joints
        if len(q) < n or len(qd) < n:
            return

        # Use only the first n joints (robot joints)
        q = q[:n]
        qd = qd[:n]

        # Calculate the PD control
        ctrl = self.controller.compute(q, qd)

        # Limit the control input to the range [-1, 1]
        ctrl = np.clip(ctrl, -1.0, 1.0)

        # Publish the command
        cmd_msg = Float64MultiArray() # control command message
        cmd_msg.data = ctrl.tolist()
        self.cmd_pub.publish(cmd_msg)

        # Publish the controller status
        at_target = self.controller.is_at_target(q, self.tolerance)
        error = self.controller.get_position_error(q)
        max_error = float(np.max(np.abs(error)))

        status_msg = String()
        status_msg.data = (
            f'target_reached={at_target} '
            f'max_error={max_error:.4f} '
            f'q={q.tolist()} '
            f'target={self.controller.q_target.tolist()}'
        )
        self.status_pub.publish(status_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PDControllerNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()