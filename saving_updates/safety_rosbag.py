#!/usr/bin/python3.8

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from sensor_msgs.msg import Range
from cargobot_msgs.msg import Safety

# CONSTANTS
V_MAX_FORWARD = 2  # m/s
V_MAX_BACKWARD = 1  # m/s
V_MAX_SPIN = 1  # rad/s

DISTANCE_STOP_FORWARD = 0.5
DISTANCE_SLOW_2_FORWARD = 4.0

DISTANCE_STOP_BACKWARD = 0.5
DISTANCE_SLOW_2_BACKWARD = 4.0

DISTANCE_STOP_SPIN = 0.1
DISTANCE_SLOW_2_SPIN = 1.0


class SafetyNode:
    """
    A ROS node to handle safety-related calculations for a car using sonar and lidar sensors.
    """

    def __init__(self):
        # Initialize sensor data
        self.safety_msg = Safety()
        self.range_sonar_forward = 0
        self.range_sonar_backward = 0
        self.range_sonar_left = 0
        self.range_sonar_right = 0
        self.range_lidar_front_right = 0
        self.range_lidar_front_center = 0
        self.range_lidar_front_left = 0

        # Old data for filtering
        self.range_lidar_front_right_old = 0.1
        self.range_lidar_front_center_old = 0.1
        self.range_lidar_front_left_old = 0.1
        self.range_sonar_backward_old = 0.1
        self.filter_sonar_forward = []

        # ROS subscribers for sensor data
        rospy.Subscriber("/sonar_front", Range, self.cb_sonar_forward)
        rospy.Subscriber("/sonar_back", Range, self.cb_sonar_backward)
        rospy.Subscriber("/sonar_left", Range, self.cb_sonar_left)
        rospy.Subscriber("/sonar_right", Range, self.cb_sonar_right)
        rospy.Subscriber("/lidar_1D_front_right", Range, self.cb_lidar_front_right)
        rospy.Subscriber("/lidar_1D_front_center", Range, self.cb_lidar_front_center)
        rospy.Subscriber("/lidar_1D_front_left", Range, self.cb_lidar_front_left)

        # ROS publisher for safety messages
        self.pub_safety = rospy.Publisher('safety_limit_speed', Safety, queue_size=10)

    def cb_sonar_forward(self, msg):
        """
        Callback for the front sonar sensor.
        Filters the sensor data using a moving average.
        """
        self.filter_sonar_forward.append(msg.range)
        if len(self.filter_sonar_forward) > 20:
            self.filter_sonar_forward.pop(0)

        self.range_sonar_forward = sum(self.filter_sonar_forward) / len(self.filter_sonar_forward)

    def cb_sonar_backward(self, msg):
        """
        Callback for the backward sonar sensor.
        Handles edge cases where the range might be zero.
        """
        if msg.range == 0.0:
            self.range_sonar_backward = self.range_sonar_backward_old if self.range_sonar_backward_old >= 0.1 else 4.0
        else:
            self.range_sonar_backward = msg.range
            self.range_sonar_backward_old = self.range_sonar_backward

    def cb_sonar_left(self, msg):
        """Callback for the left sonar sensor."""
        self.range_sonar_left = msg.range

    def cb_sonar_right(self, msg):
        """Callback for the right sonar sensor."""
        self.range_sonar_right = msg.range

    def cb_lidar_front_right(self, msg):
        """
        Callback for the front-right lidar sensor.
        Handles invalid ranges (e.g., 0 or 22).
        """
        if msg.range in [0.0, 22.0]:
            self.range_lidar_front_right = self.range_lidar_front_right_old if self.range_lidar_front_right_old >= 0.1 else 4.0
        else:
            self.range_lidar_front_right = msg.range
            self.range_lidar_front_right_old = self.range_lidar_front_right

    def cb_lidar_front_center(self, msg):
        """Callback for the front-center lidar sensor."""
        if msg.range in [0.0, 22.0]:
            self.range_lidar_front_center = self.range_lidar_front_center_old if self.range_lidar_front_center_old >= 0.1 else 4.0
        else:
            self.range_lidar_front_center = msg.range
            self.range_lidar_front_center_old = self.range_lidar_front_center

    def cb_lidar_front_left(self, msg):
        """Callback for the front-left lidar sensor."""
        if msg.range in [0.0, 22.0]:
            self.range_lidar_front_left = self.range_lidar_front_left_old if self.range_lidar_front_left_old >= 0.1 else 4.0
        else:
            self.range_lidar_front_left = msg.range
            self.range_lidar_front_left_old = self.range_lidar_front_left

    def calculate_speed_limits(self):
        """
        Calculates speed limits based on sensor data and publishes a safety message.
        """
        self.safety_msg.data_sensor = [
            self.range_sonar_right,
            self.range_lidar_front_right if self.range_lidar_front_right else 10.0,
            self.range_sonar_forward,
            self.range_lidar_front_center if self.range_lidar_front_center else 10.0,
            self.range_lidar_front_left if self.range_lidar_front_left else 10.0,
            self.range_sonar_left,
            self.range_sonar_backward
        ]

        # Forward speed limit
        forward_sensors = [self.safety_msg.data_sensor[1], self.safety_msg.data_sensor[4]]
        if min(self.filter_invalid_values(forward_sensors)) < DISTANCE_STOP_FORWARD:
            adjusted_distance = 2 * min(self.filter_invalid_values(forward_sensors))
            forward_data = [self.safety_msg.data_sensor[2], self.safety_msg.data_sensor[3], adjusted_distance]
        else:
            forward_data = [self.safety_msg.data_sensor[2], self.safety_msg.data_sensor[3]]

        self.safety_msg.limit_forward_speed = self.calculate_speed(forward_data, DISTANCE_STOP_FORWARD, DISTANCE_SLOW_2_FORWARD, V_MAX_FORWARD)

        # Backward speed limit
        self.safety_msg.limit_backward_speed = self.calculate_speed(
            [self.safety_msg.data_sensor[6]], DISTANCE_STOP_BACKWARD, DISTANCE_SLOW_2_BACKWARD, V_MAX_BACKWARD
        )

        # Spin speed limits
        self.safety_msg.limit_spin_right_speed = self.calculate_speed(
            self.safety_msg.data_sensor[0:4], DISTANCE_STOP_SPIN, DISTANCE_SLOW_2_SPIN, V_MAX_SPIN
        )
        self.safety_msg.limit_spin_left_speed = self.calculate_speed(
            self.safety_msg.data_sensor[2:5], DISTANCE_STOP_SPIN, DISTANCE_SLOW_2_SPIN, V_MAX_SPIN
        )

        self.pub_safety.publish(self.safety_msg)

    @staticmethod
    def filter_invalid_values(array):
        """Replaces invalid sensor readings with default values."""
        return [10.0 if x == 0.0 else (0 if x == 22.0 else x) for x in array]

    @staticmethod
    def calculate_speed(sensors, distance_stop, distance_slow_2, v_max):
        """
        Calculates the speed limit based on sensor distances.
        """
        min_range = min(sensors)
        if 0.0 <= min_range < distance_stop:
            return 0
        elif distance_stop < min_range < distance_slow_2:
            return ((min_range - distance_stop) / (distance_slow_2 - distance_stop)) * v_max
        else:
            return v_max

    def main_loop(self):
        """Main loop to continuously calculate speed limits."""
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            self.calculate_speed_limits()
            rate.sleep()


if __name__ == '__main__':
    rospy.init_node('safety_node')
    safety_node = SafetyNode()
    safety_node.main_loop()
