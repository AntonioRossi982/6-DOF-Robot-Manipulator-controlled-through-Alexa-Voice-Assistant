import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration


class TaskPlanner(Node):
    def __init__(self):
        super().__init__('task_planner')

        # Define action clients for robot arm and gripper
        self.arm_client = ActionClient(self, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory')
        self.gripper_client = ActionClient(self, FollowJointTrajectory, '/gripper_controller/follow_joint_trajectory')

        # Arm positions in radians
        self.arm_positions = {
            'home':  [0.0,  0.0,   0.0],
            'pick':  [0.8,  0.5,  -0.5],
            'place': [-0.8, 0.5,  -0.5],
        }

        # Gripper positions
        self.GRIPPER_OPEN   = 0.0
        self.GRIPPER_CLOSED = -1.2

        # State sequence
        self.states = [
            ('arm',     'home',          2),
            ('gripper', 'open',          1),
            ('arm',     'pick',          2),
            ('gripper', 'close',         1),
            ('arm',     'place',         2),
            ('gripper', 'open',          1),
            ('arm',     'home',          2),
        ]
        self.current_state = 0

        self.get_logger().info('Task Planner started — Waiting for action servers...')

        # Wait for action servers
        self.arm_client.wait_for_server()
        self.gripper_client.wait_for_server()

        self.get_logger().info('Action servers ready — starting sequence')

        # Start first state
        self.execute_next_state()

    def execute_next_state(self):
        if self.current_state >= len(self.states):
            self.get_logger().info('Sequence completed')
            return

        controller, action, duration = self.states[self.current_state]
        self.get_logger().info(
            f'State {self.current_state + 1}/{len(self.states)}: '
            f'{controller} → {action}'
        )

        if controller == 'arm':
            self.send_arm_goal(self.arm_positions[action], duration)
            
        elif controller == 'gripper':
            position = (
                self.GRIPPER_OPEN
                if action == 'open'
                else self.GRIPPER_CLOSED
            )
            self.send_gripper_goal(position, duration)

    def send_arm_goal(self, positions, duration_sec):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ['joint_1', 'joint_2', 'joint_3']
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = Duration(sec=duration_sec)
        goal.trajectory.points = [point]

        self.get_logger().info(f'Arm → {positions}')
        send_future = self.arm_client.send_goal_async(goal)
        send_future.add_done_callback(self.goal_response_callback)

    def send_gripper_goal(self, position, duration_sec):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ['joint_4']
        point = JointTrajectoryPoint()
        point.positions = [position]
        point.time_from_start = Duration(sec=duration_sec)
        goal.trajectory.points = [point]

        state = 'OPEN' if position >= 0.0 else 'CLOSED'
        self.get_logger().info(f'Gripper → {state} ({position:.2f} rad)')
        send_future = self.gripper_client.send_goal_async(goal)
        send_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error('Goal refused')
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        result = future.result()

        if result.status == 4:  # SUCCEEDED
            self.get_logger().info('Goal completed')
        else:
            self.get_logger().warn(f'Goal terminated with status: {result.status}')

        # Move to next state
        self.current_state += 1
        self.execute_next_state()


def main(args=None):
    rclpy.init(args=args)
    node = TaskPlanner()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()