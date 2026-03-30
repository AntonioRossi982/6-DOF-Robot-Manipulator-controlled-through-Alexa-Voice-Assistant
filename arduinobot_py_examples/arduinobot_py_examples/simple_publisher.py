import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class SimplePublisher(Node): 
    def __init__(self):
        super().__init__("first_publisher")
        self.pub_ = self.create_publisher(String, "Chatter", 10)
        self.counter_ = 0 
        self.frequency = 1.0
        self.get_logger(). info("Publishing at %d Hz" % self.frequency)
        self.timer_ = self.create_timer(self.frequency, self.timerCallback)

    def timerCallback(self):
        msg = String()
        msg.data = "Hey - counter: %d" % self.counter_ 
        self.pub_.publish(msg)
        self.counter_ += 1

def main():
    rclpy.init()
    first_publisher = SimplePublisher()
    rclpy.spin(first_publisher)
    first_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
