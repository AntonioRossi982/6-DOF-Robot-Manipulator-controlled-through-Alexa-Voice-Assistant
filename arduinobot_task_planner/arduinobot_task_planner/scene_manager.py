import rclpy
from rclpy.node import Node
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose


class SceneManager(Node):
    def __init__(self):
        super().__init__('scene_manager')

        self.collision_pub = self.create_publisher(CollisionObject, '/collision_object', 10)

        self.publish_count = 0
        self.max_publishes = 5

        self.get_logger().info('Waiting for MoveIt2 — 10 seconds to start...')

        # Wait 10 seconds before starting
        self.create_timer(10.0, self.start_publishing)

    def start_publishing(self):
        # Publish every 2 seconds, 5 times
        self.publish_timer = self.create_timer(2.0, self.publish_objects)

    def publish_objects(self):
        if self.publish_count >= self.max_publishes:
            self.publish_timer.cancel()
            self.get_logger().info('Publishing completed')
            return

        self.publish_count += 1
        self.get_logger().info(
            f'Publishing {self.publish_count}/{self.max_publishes}'
        )
        self.add_table()
        self.add_box()

    def add_table(self):
        obj = CollisionObject()
        obj.id = 'table'
        obj.header.frame_id = 'world'
        obj.header.stamp = self.get_clock().now().to_msg()

        shape = SolidPrimitive()
        shape.type = SolidPrimitive.BOX
        shape.dimensions = [0.6, 0.6, 0.05]

        pose = Pose()
        pose.position.x = 0.0
        pose.position.y = 0.0
        pose.position.z = -0.05
        pose.orientation.w = 1.0

        obj.primitives = [shape]
        obj.primitive_poses = [pose]
        obj.operation = CollisionObject.ADD

        self.collision_pub.publish(obj)

    def add_box(self):
        obj = CollisionObject()
        obj.id = 'obstacle'
        obj.header.frame_id = 'world'
        obj.header.stamp = self.get_clock().now().to_msg()

        shape = SolidPrimitive()
        shape.type = SolidPrimitive.BOX
        shape.dimensions = [0.05, 0.05, 0.3]

        pose = Pose()
        pose.position.x = 0.15
        pose.position.y = 0.15
        pose.position.z = 0.15
        pose.orientation.w = 1.0

        obj.primitives = [shape]
        obj.primitive_poses = [pose]
        obj.operation = CollisionObject.ADD

        self.collision_pub.publish(obj)


def main(args=None):
    rclpy.init(args=args)
    node = SceneManager()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()