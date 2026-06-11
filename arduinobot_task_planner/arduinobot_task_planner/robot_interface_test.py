import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import threading
import time


class RobotInterface(Node):
    def __init__(self):
        super().__init__('robot_interface')

        self.arm_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory'
        )

        self.gripper_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/gripper_controller/follow_joint_trajectory'
        )

        self.arm_presets = {
            'home':  [0.0,  0.0,   0.0],
            'pick':  [0.8,  0.5,  -0.5],
            'place': [-0.8, 0.5,  -0.5],
        }

        self.GRIPPER_OPEN   = 0.0
        self.GRIPPER_CLOSED = -1.2

        self.is_executing = False
        self._pending_duration = 1.0

        self.get_logger().info('Waiting for action servers...')
        self.arm_client.wait_for_server()
        self.gripper_client.wait_for_server()
        self.get_logger().info('Action servers ready')

        self.print_help()

        self.input_thread = threading.Thread(
            target=self.input_loop,
            daemon=True
        )
        self.input_thread.start()

    def print_help(self):
        print('\n============================')
        print('   Arduinobot Interface')
        print('============================')
        print('  home   — home position')
        print('  pick   — pick position')
        print('  place  — place position')
        print('  open   — open gripper')
        print('  close  — close gripper')
        print('  task   — pick and place sequence')
        print('  help   — show this menu')
        print('  quit   — exit')
        print('============================\n')

    def send_arm(self, positions, duration_sec=2):
        if self.is_executing:
            print('Robot moving — wait...')
            return

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ['joint_1', 'joint_2', 'joint_3']
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = Duration(sec=duration_sec)
        goal.trajectory.points = [point]

        self._pending_duration = float(duration_sec)
        self.is_executing = True
        future = self.arm_client.send_goal_async(goal)
        future.add_done_callback(self.goal_response_callback)
        self.get_logger().info(f'Arm → {positions}')

    def send_gripper(self, position, duration_sec=2):
        if self.is_executing:
            print('Robot moving — wait...')
            return

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ['joint_4']
        point = JointTrajectoryPoint()
        point.positions = [position]
        point.time_from_start = Duration(sec=duration_sec)
        goal.trajectory.points = [point]

        self._pending_duration = float(duration_sec)
        self.is_executing = True
        future = self.gripper_client.send_goal_async(goal)
        future.add_done_callback(self.goal_response_callback)
        state = 'OPEN' if position >= 0.0 else 'CLOSED'
        self.get_logger().info(f'Gripper → {state}')

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected')
            self.is_executing = False
            return
        self.get_logger().info('Goal accepted')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        # Attendi la durata effettiva del movimento prima di sbloccare
        time.sleep(self._pending_duration)
        self.is_executing = False
        self.get_logger().info('Movement completed')

    def execute_task(self):
        """
        Sequenza corretta:
        1. Vai in posizione pick
        2. Apri gripper
        3. Chiudi gripper
        4. Vai in posizione place
        5. Apri gripper
        6. Chiudi gripper
        7. Torna in posizione home
        """
        self.get_logger().info('Starting pick and place sequence...')

        steps = [
            ('arm',     self.arm_presets['pick'],  2),
            ('gripper', self.GRIPPER_OPEN,          2),
            ('gripper', self.GRIPPER_CLOSED,        2),
            ('arm',     self.arm_presets['place'],  2),
            ('gripper', self.GRIPPER_OPEN,          2),
            ('gripper', self.GRIPPER_CLOSED,        2),
            ('arm',     self.arm_presets['home'],   2),
        ]

        for controller, target, duration in steps:
            while self.is_executing:
                time.sleep(0.1)

            if controller == 'arm':
                self.send_arm(target, duration)
            else:
                self.send_gripper(target, duration)

            time.sleep(duration + 0.5)

        while self.is_executing:
            time.sleep(0.1)

        self.get_logger().info('Pick and place sequence completed')

    def input_loop(self):
        while True:
            try:
                cmd = input('\n> ').strip().lower()
            except EOFError:
                break

            if cmd in self.arm_presets:
                self.send_arm(self.arm_presets[cmd])

            elif cmd == 'open':
                self.send_gripper(self.GRIPPER_OPEN)

            elif cmd == 'close':
                self.send_gripper(self.GRIPPER_CLOSED)

            elif cmd == 'task':
                if self.is_executing:
                    print('Robot moving — wait...')
                else:
                    t = threading.Thread(
                        target=self.execute_task,
                        daemon=True
                    )
                    t.start()

            elif cmd == 'help':
                self.print_help()

            elif cmd == 'quit':
                self.get_logger().info('Exiting program')
                break

            else:
                print(f'Unknown command: "{cmd}"')
                print('Type "help" for available commands')


def main(args=None):
    rclpy.init(args=args)
    node = RobotInterface()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()