import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time


class MissionPlannerNode(Node):
    """
    ROS2 node for mission planning.
    
    Publishes a sequence of targets to /target_position.
    Waits for the controller to confirm that the target has been reached
    before sending a new one.
    
    This node simulates a high-level system that
    controls the robot without knowing anything about the physics
    or the type of controller used.
    """

    def __init__(self):
        super().__init__('mission_planner')

        # Sequence of targets — format "j1,j2" in radians
        self.targets = [
            "0.0,0.0,0.0,0.0",        # home
            "0.8,0.5,-0.3,0.0",       # first position — arm went to the right
            "-0.8,0.5,-0.3,0.0",      # second position — arm went to the left
            "0.0,0.8,0.0,-0.8",       # third position — arm up, gripper close
            "0.5,-0.3,0.5,0.0",       # fourth position
            "0.0,0.0,0.0,0.0",        # home
        ]

        self.current_target_idx = 0 # index of the current target in the list
        self.target_reached = False
        self.time_on_target = 0.0
        self.min_time_on_target = 2.0  # seconds

        # Target publisher (topic: /target_position)
        self.target_pub = self.create_publisher(String, '/target_position', 10)

        # Subscriber for the controller status (topic: /controller_status)
        self.status_sub = self.create_subscription(String, '/controller_status', self.status_callback, 10)

        # Timer for the mission logic — 10 Hz (frequency)
        self.mission_timer = self.create_timer(0.1, self.mission_step)

        # Send the first target after 2 seconds
        # Gives time for the simulator to start
        self.startup_timer = self.create_timer(2.0, self.send_first_target)
        self.get_logger().info('Mission planner ready')

    def send_first_target(self):
        """Send the first target and cancel the startup timer."""
        self.startup_timer.cancel()
        self.send_current_target()

    def send_current_target(self):
        """Publish the current target on topic: /target_position."""
        target = self.targets[self.current_target_idx]
        msg = String()
        msg.data = target
        self.target_pub.publish(msg)
        self.get_logger().info(f'Target [{self.current_target_idx}]: {target} rad')

    def status_callback(self, msg):
        """
        Callback for the controller status.
        Extracts the target_reached flag from the status message.
        """
        try:
            # Parsing of the status message
            # Format: "target_reached=True -> "target_reached" : "True" and
            # max_error=0.001 -> "max_error" : "0.001"
            parts = dict(
                p.split('=') for p in msg.data.split()
                if '=' in p
            )
            # read the value in the dictionary
            self.target_reached = parts.get(
                'target_reached', 'False'
            ) == 'True'
        except Exception:
            pass

    def mission_step(self):
        """
        Mission logic — executed at 10 Hz.
        Moves to the next target when:
        1. The controller has reached the current target
        2. The robot has stayed on the target for min_time_on_target (in seconds)
        """
        if self.target_reached: 
            # update the time on target only if the target is reached
            self.time_on_target += 0.1  # 0.1s per step at 10 Hz
        else:
            # reset if the target is not reached or lost
            self.time_on_target = 0.0

        # if the target is reached and the robot has stayed on it for the required time, like 2 seconds, go to the next target
        if (self.target_reached and self.time_on_target >= self.min_time_on_target):
            # Go to the next target in the mission sequence
            self.current_target_idx = ((self.current_target_idx + 1) % len(self.targets))
            self.time_on_target = 0.0
            self.target_reached = False
            self.send_current_target()


def main(args=None):
    rclpy.init(args=args)
    node = MissionPlannerNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()