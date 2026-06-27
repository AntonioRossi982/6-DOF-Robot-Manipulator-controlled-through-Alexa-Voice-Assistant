import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray
import mujoco
import mujoco.viewer
import numpy as np
import os
import threading
import time


class MuJoCoSimNode(Node):
    """
    ROS2 node that wraps MuJoCo as a physics simulator.
    
    Publishes:
    - /mujoco/joint_states (sensor_msgs/JointState)
      Positions and velocities of all robot joints
    - /mujoco/ee_pose (geometry_msgs/PoseStamped)  
      Position of the end effector in the world frame
      
    Subscribes to:
    - /mujoco/joint_commands (std_msgs/Float64MultiArray)
      Control signals for the actuators
      
    The node runs at 500 Hz (MuJoCo timestep = 0.002s)
    and publishes the joint states at every step.
    """

    def __init__(self):
        super().__init__('mujoco_sim')

        # Model Path (2 DOF simple planar arm) - change the path if you want to use a different model
        model_path = os.path.expanduser('~/mujoco_robotics/models/arduinobot.xml')

        # Load the MuJoCo model and create a data struct for simulation
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # # Lock to access thread-safe data 
        # # Necessary because ROS2 and MuJoCo viewer run in different threads
        # self.data_lock = threading.Lock()

        # Joint names extracted from the model
        # Build the JointState message with the extracted names
        self.joint_names = []
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            self.joint_names.append(name)

        self.get_logger().info(f'Model loaded: {self.model.njnt} joints, ' f'{self.model.nu} actuators')
        self.get_logger().info(f'Joints: {self.joint_names}')

        # Joint Publisher — publishes joint states at 500 Hz
        # QoS depth=10: get the last 10 messages in a queue
        self.joint_state_pub = self.create_publisher(JointState, '/mujoco/joint_states', 10)

        # Publisher for end effector pose
        self.ee_pose_pub = self.create_publisher(PoseStamped, '/mujoco/ee_pose', 10)

        # Subscriber for control commands
        # When a message arrives, it calls the associated function self.command_callback
        self.cmd_sub = self.create_subscription(Float64MultiArray, '/mujoco/joint_commands', self.command_callback, 10)

        # Timer for the simulation loop
        # period = MuJoCo time step = 0.002s = 500 Hz
        # This is the heart of the node — it is called at every simulation step
        
        # sim_period = self.model.opt.timestep
        # self.sim_timer = self.create_timer(sim_period, self.simulation_step)
        self.sim_timer = self.create_timer(self.model.opt.timestep, self.simulation_step)
        self.get_logger().info('Simulator ready')

        # self.get_logger().info(f'Simulator started at {1/sim_period:.0f} Hz')

    def command_callback(self, msg):
        """
        Callback called when a command arrives at /mujoco/joint_commands.
        Writes the received values to data.ctrl — the vector of
        control signals that MuJoCo applies to the actuators
        at the next simulation step.
        
        Args:
            msg: Float64MultiArray containing the control signals
            len(msg.data) must be equal to model.nu
        """
        if len(msg.data) != self.model.nu:
            self.get_logger().warn(f'Command with {len(msg.data)} values ' f'but the robot has {self.model.nu} actuators')
            return

        # Write the commands to the MuJoCo control vector
        # The clip function ensures that the values stay within the limits
        self.data.ctrl[:] = np.clip(
            msg.data,
            self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1]
        )

    def simulation_step(self):
        """
        Timer callback — executed at every timestep.
        Sequence:
        1. Advance the simulation by one timestep
        2. Publish the joint states
        3. Publish the end-effector position
        """
        # Advance the simulation
        # Apply data.ctrl, calculate forces, integrate, update positions
        mujoco.mj_step(self.model, self.data)

        # Publish joint states
        self.publish_joint_states()

        # Publish end-effector position
        self.publish_ee_pose()

    def publish_joint_states(self):
        """
        Publish the joint positions and velocities on /mujoco/joint_states.
        The JointState message is the standard ROS2 message for communicating
        the state of the joints — used by RViz2, MoveIt2 and other tools.
        """
        msg = JointState() # msg is an istance of sensor_msgs/JointState

        # Header with current timestamp
        # Fundamental for synchronization between nodes
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'

        # Names of the joints
        msg.name = self.joint_names

        # Positions in radians — from data.qpos
        msg.position = self.data.qpos.tolist()

        # Velocities in rad/s — from data.qvel
        msg.velocity = self.data.qvel.tolist()

        # Effort (forces/torques) — from data.actuator_force
        # Convert to list for ROS2 serialization
        msg.effort = self.data.actuator_force.tolist()

        self.joint_state_pub.publish(msg)

    def publish_ee_pose(self):
        """
        Publish the position of the end effector on /mujoco/ee_pose.
        Uses the body 'end_effector' defined in the MJCF.
        data.xpos[body_id] contains the xyz position in the world frame
        after the last mj_step.
        """
        try:
            ee_id = self.model.body('end_effector').id
        except Exception:
            return

        msg = PoseStamped() # msg is an istance of geometry_msgs/PoseStamped
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'

        # Position xyz of the end effector
        pos = self.data.xpos[ee_id]
        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])

        # Orientation as quaternion
        # data.xquat[body_id] = [w, x, y, z] in MuJoCo
        # ROS2 uses [x, y, z, w] — note the different order
        quat = self.data.xquat[ee_id]
        msg.pose.orientation.w = float(quat[0])
        msg.pose.orientation.x = float(quat[1])
        msg.pose.orientation.y = float(quat[2])
        msg.pose.orientation.z = float(quat[3])

        self.ee_pose_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MuJoCoSimNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()