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
            'home':  [0.0,  0.0,  0.0],
            'pick':  [0.8,  0.5, -0.5],
            'place': [-0.8, 0.5, -0.5],
        }

        # 0.0 represents the closed gripper (starting position), 
        # while -1.45 represents the open gripper (close to -pi/2 which is the physical limit of the gripper)
        self.GRIPPER_CLOSED = 0.0
        self.GRIPPER_OPEN = -1.45

        self.task_running = False

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

    # ==========================================================
    # HELP
    # ==========================================================

    def print_help(self):
        print('\n============================')
        print('   Arduinobot Interface')
        print('============================')
        print('  home   - home position')
        print('  pick   - pick position')
        print('  place  - place position')
        print('  open   - open gripper')
        print('  close  - close gripper')
        print('  task   - pick and place')
        print('  help   - show menu')
        print('  quit   - exit')
        print('============================\n')

    # ==========================================================
    # ACTION EXECUTION
    # ==========================================================

    def execute_action_and_wait(self, client, joint_names, positions, duration_sec):
        goal = FollowJointTrajectory.Goal()

        goal.trajectory.joint_names = joint_names

        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = Duration(
            sec=int(duration_sec)
        )

        goal.trajectory.points = [point]

        send_future = client.send_goal_async(goal)

        while not send_future.done():
            time.sleep(0.01)

        goal_handle = send_future.result()

        if goal_handle is None:
            self.get_logger().error(
                'Failed to communicate with action server'
            )
            return False

        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected')
            return False

        self.get_logger().info('Goal accepted')

        result_future = goal_handle.get_result_async()

        while not result_future.done():
            time.sleep(0.01)

        result = result_future.result()

        if result is None:
            self.get_logger().error(
                'Failed to receive action result'
            )
            return False

        self.get_logger().info('Movement completed')

        return True

    # ==========================================================
    # ARM
    # ==========================================================

    def move_arm(self, positions, duration_sec=2):

        self.get_logger().info(
            f'Arm -> {positions}'
        )

        return self.execute_action_and_wait(
            self.arm_client,
            ['joint_1', 'joint_2', 'joint_3'],
            positions,
            duration_sec
        )

    # ==========================================================
    # GRIPPER
    # ==========================================================

    def move_gripper(self, position, duration_sec=2):

        state = (
            'OPEN'
            if position == self.GRIPPER_OPEN
            else 'CLOSED'
        )

        self.get_logger().info(
            f'Gripper -> {state}'
        )

        return self.execute_action_and_wait(
            self.gripper_client,
            ['joint_4'],
            [position],
            duration_sec
        )

    # ==========================================================
    # PICK AND PLACE
    # ==========================================================

    def execute_task(self):

        if self.task_running:
            self.get_logger().warning(
                'Task already running'
            )
            return

        self.task_running = True

        self.get_logger().info(
            'Starting pick and place sequence...'
        )

        try:

            # 1) PICK
            if not self.move_arm(
                self.arm_presets['pick']
            ):
                return

            # 2) OPEN
            if not self.move_gripper(
                self.GRIPPER_OPEN
            ):
                return

            # 3) CLOSE (pick object)
            if not self.move_gripper(
                self.GRIPPER_CLOSED
            ):
                return

            # 4) PLACE
            if not self.move_arm(
                self.arm_presets['place']
            ):
                return

            # 5) OPEN (release object)
            if not self.move_gripper(
                self.GRIPPER_OPEN
            ):
                return

            # 6) CLOSE
            if not self.move_gripper(
                self.GRIPPER_CLOSED
            ):
                return

            # 7) HOME
            if not self.move_arm(
                self.arm_presets['home']
            ):
                return

            self.get_logger().info(
                'Pick and place sequence completed'
            )

        finally:
            self.task_running = False

    # ==========================================================
    # INPUT LOOP
    # ==========================================================

    def input_loop(self):

        while rclpy.ok():

            try:
                cmd = input('\n> ').strip().lower()

            except EOFError:
                break

            if cmd in self.arm_presets:

                threading.Thread(
                    target=self.move_arm,
                    args=(self.arm_presets[cmd],),
                    daemon=True
                ).start()

            elif cmd == 'open':

                threading.Thread(
                    target=self.move_gripper,
                    args=(self.GRIPPER_OPEN,),
                    daemon=True
                ).start()

            elif cmd == 'close':

                threading.Thread(
                    target=self.move_gripper,
                    args=(self.GRIPPER_CLOSED,),
                    daemon=True
                ).start()

            elif cmd == 'task':

                threading.Thread(
                    target=self.execute_task,
                    daemon=True
                ).start()

            elif cmd == 'help':

                self.print_help()

            elif cmd == 'quit':

                self.get_logger().info(
                    'Exiting program'
                )
                break

            else:

                print(
                    f'Unknown command: "{cmd}"'
                )
                print(
                    'Type "help" for available commands'
                )


def main(args=None):

    rclpy.init(args=args)
    node = RobotInterface()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()