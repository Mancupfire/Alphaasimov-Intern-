#!/usr/bin/python3.8
import re
from tabnanny import check
import time
import traceback
import os
import datetime
import math
import rospy
import roslib
import actionlib
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from sensor_msgs.msg import Range
from cargobot_msgs.msg import Safety 
#import interfaces.log as log 

# DEFINES


V_MAX_FORWARD = 2 # m/s
V_MAX_BACKWARD = 1  # m/s
V_MAX_SPIN = 1 #rad/s

DISTANCE_STOP_FORWARD = 0.5
DISTANCE_SLOW_2_FORWARD = 4.0

DISTANCE_STOP_BACKWARD = 0.5
DISTANCE_SLOW_2_BACKWARD = 4.0

DISTANCE_STOP_SPIN = 0.1
DISTANCE_SLOW_2_SPIN = 1.0


# > The Safety class is a container for the safety-related data of a car
class Safety_():
    """"""
    
    def __init__(self):

        """
        A constructor.
        """
        self.safety_msg=Safety()
        self.range_sonar_forward = 0
        self.sonar_backward = 0
        self.range_sonar_left = 0
        self.range_sonar_right = 0
        self.range_lidar_1D_front_right = 0
        self.range_lidar_1D_front_center= 0
        self.range_lidar_1D_front_left  = 0

        self.range_lidar_1D_front_right_old = 0.1
        self.range_lidar_1D_front_center_old = 0.1
        self.range_lidar_1D_front_left_old  = 0.1
        self.sonar_backward_old = 0.1
        self.sonar_forward_old = 0.1
        self.range_sonar_forward_temp=0
    
        self.filter_sonar_fw = [] 
        rospy.Subscriber("/sonar_front", Range, self.cb_status_sonar_forward)
        rospy.Subscriber("/sonar_back", Range, self.cb_status_lidar_backward)
        rospy.Subscriber("/sonar_left", Range, self.cb_status_sonar_left)
        rospy.Subscriber("/sonar_right", Range, self.cb_status_sonar_right)
        rospy.Subscriber("/lidar_1D_front_right", Range, self.cb_status_lidar_1D_front_right)
        rospy.Subscriber("/lidar_1D_front_center", Range, self.cb_status_lidar_1D_front_center)
        rospy.Subscriber("/lidar_1D_front_left", Range, self.cb_status_lidar_1D_front_left)
 
        self.pub_safety= rospy.Publisher('safety_limit_speed', Safety, queue_size=10)

        
    def cb_status_sonar_forward(self,msg):
        """
        This function is a callback function that is called when the sonar forward sensor publishes a
        message
        
        :param msg: The message that was received
        """
        self.range_sonar_forward_temp = msg.range
        # if  abs(self.range_sonar_forward_temp-self.range_sonar_forward) >6.5:
        #     self.range_sonar_forward_temp = self.range_sonar_forward
        
        self.filter_sonar_fw.append(self.range_sonar_forward_temp)
        
        if len(self.filter_sonar_fw)>20:
            self.filter_sonar_fw.pop(0)

        self.range_sonar_forward=sum(self.filter_sonar_fw) / len(self.filter_sonar_fw)
        #self.range_sonar_forward=min(self.filter_sonar_fw)
        # if  self.range_sonar_forward == self.sonar_forward_old:
        #     self.range_sonar_forward = 4.0
        # self.sonar_forward_old = msg.range
        self.range_sonar_forward = msg.range

        

       # print("sonar_forward",msg.range)
        #log.loginfo("sonar_forward %s",self.range_sonar_forward)

    def cb_status_lidar_backward(self,msg):
        
        """
        This function is a callback function that is called when the sonar sensor on the back of the
        robot detects an object
        
        :param msg: The message that was received
        """

        if msg.range==0.0:
            if self.sonar_backward_old<0.1:
                self.sonar_backward =   self.sonar_backward_old
            else:
               self.sonar_backward=4

        else:
            self.sonar_backward=msg.range
            self.sonar_backward_old =  self.sonar_backward
        #print("lidar_backward",msg.range)
        #self.sonar_backward = msg.range
        #log.loginfo("sonar_backward %s", self.sonar_backward)
        
    def cb_status_sonar_left(self,msg):
        """
        This function is a callback function that is called when the sonar sensor on the left side of the
        robot detects an object
        
        :param msg: the message that was received
        """
        
        self.range_sonar_left=msg.range
        #ok print("sonar_left",msg.range)
    def cb_status_sonar_right(self,msg):
        """
        This function is a callback function that is called when the sonar sensor on the right side of
        the robot detects an object
        
        :param msg: The message that was received
        """
       
        self.range_sonar_right=msg.range
        # ok print("sonar_right",msg.range)
    def cb_status_lidar_1D_front_right(self,msg):
        """
        This function is a callback function that is called when the topic /scan is published
        
        :param msg: the message that is received from the topic
        """

        if msg.range==0.0 or msg.range==22.0:
            if self.range_lidar_1D_front_right_old<0.1:
                self.range_lidar_1D_front_right =  self.range_lidar_1D_front_right_old
            else:
                self.range_lidar_1D_front_right=4 

        else:
            self.range_lidar_1D_front_right=msg.range
            self.range_lidar_1D_front_right_old = self.range_lidar_1D_front_right

        # ok print("lidar_1D_front_right",msg.range)
        
    def cb_status_lidar_1D_front_center(self,msg):
        """
        This function is a callback function that is called when a message is received on the topic
        /scan. The function takes the message and stores the data in the class variable
        self.lidar_1D_front_center
        
        :param msg: the message that was received
        """
        if msg.range==0.0 or msg.range==22.0:
            if self.range_lidar_1D_front_center_old<0.1:
                self.range_lidar_1D_front_center =   self.range_lidar_1D_front_center_old
            else:
               self.range_lidar_1D_front_center=4

        else:
            self.range_lidar_1D_front_center=msg.range
            self.range_lidar_1D_front_center_old =  self.range_lidar_1D_front_center

        #self.range_lidar_1D_front_center=msg.range

        #ok print("lidar_1D_front_center",msg.range)
    def cb_status_lidar_1D_front_left(self,msg):
        
        """
        This function is a callback function that is called when a message is received on the topic
        /scan. The function takes the message and stores the data in the class variable
        self.lidar_1D_front_left
        
        :param msg: the message that is received from the topic
        """
        if msg.range==0.0 or msg.range==22.0:
            if self.range_lidar_1D_front_left<0.1:
                self.range_lidar_1D_front_left =   self.range_lidar_1D_front_left_old
            else:
                self.range_lidar_1D_front_left=4
        else:
            self.range_lidar_1D_front_left=msg.range
            self.range_lidar_1D_front_left_old =  self.range_lidar_1D_front_left

        #self.range_lidar_1D_front_left=msg.range
        #print("lidar_1D_front_left",msg.range)
    # 
    def caculator_speed_car(self):
     
        # self.safety_msg.data_sensor = [
        # self.range_sonar_right,
        # self.range_lidar_1D_front_right if self.range_lidar_1D_front_right  else 10.0, 
        # self.range_sonar_forward,
        # self.range_lidar_1D_front_center if self.range_lidar_1D_front_center  else 10.0,
        # self.range_lidar_1D_front_left if self.range_lidar_1D_front_left  else 10.0,
        # self.range_sonar_left,
        # self.sonar_backward]
        
        self.safety_msg.data_sensor = [
        self.range_sonar_right,
        self.range_lidar_1D_front_right if self.range_lidar_1D_front_right  else 10.0, 
        self.range_sonar_forward,
        self.range_lidar_1D_front_center if self.range_lidar_1D_front_center  else 10.0,
        self.range_lidar_1D_front_left if self.range_lidar_1D_front_left  else 10.0,
        self.range_sonar_left,
        self.sonar_backward]
        # caculetor limit_forward_speed
        a=[self.safety_msg.data_sensor[1],self.safety_msg.data_sensor[4]]
        #a=[self.safety_msg.data_sensor[2] if round(self.safety_msg.data_sensor[2],2) !=0.56 else 4 ,self.safety_msg.data_sensor[1],self.safety_msg.data_sensor[4]]

        if min(self.check_array_ss(a))<0.5:
            c = 2*min(self.check_array_ss(a))
            array = [self.safety_msg.data_sensor[2],self.safety_msg.data_sensor[3],c]
        else:
            array = [self.safety_msg.data_sensor[2],self.safety_msg.data_sensor[3]]
        self.safety_msg.limit_forward_speed = self.caculator_speed(array,DISTANCE_STOP_FORWARD,DISTANCE_SLOW_2_FORWARD,V_MAX_FORWARD)

        # caculetor limit_backward_speed
        self.safety_msg.limit_backward_speed = self.caculator_speed([self.safety_msg.data_sensor[6]],DISTANCE_STOP_BACKWARD,DISTANCE_SLOW_2_BACKWARD,V_MAX_BACKWARD)

        # caculetor limit_spin_right_speed
        self.safety_msg.limit_spin_right_speed = self.caculator_speed(self.safety_msg.data_sensor[0:4],DISTANCE_STOP_SPIN,DISTANCE_SLOW_2_SPIN,V_MAX_SPIN)

        # caculetor limit_spin_left_speed
        self.safety_msg.limit_spin_left_speed = self.caculator_speed(self.safety_msg.data_sensor[2:5],DISTANCE_STOP_SPIN,DISTANCE_SLOW_2_SPIN,V_MAX_SPIN)

        self.pub_safety.publish(self.safety_msg)
        #log.loginfo("(self.safety_msg %s",self.safety_msg)

    def check_min_range(self,array):
        array_s=[]
        for a in array:
            if a== 22.0:
                array_s.append(0)
            else:
                array_s.append(a)

        min_range=array_s[0]
        for ss in array_s:
                if min_range>ss:
                    min_range=ss
        return min_range

    def check_array_ss(self,array):
        c=[]
        d=0
        for d in array:
            if d==0.0:
                c.append(10.0)
            elif d==22.0:
                c.append(0)
            else:
                c.append(d)
        
        return c
    
    def caculator_speed(self,sensors, distance_stop,distance_slow_2,v_max):
        
        limit_speed = 0
        min_range=self.check_min_range(sensors)
        if 0.0 <= min_range and min_range < distance_stop:
            limit_speed = 0
        elif distance_stop<min_range and min_range<distance_slow_2:
             limit_speed = ((min_range-distance_stop) / (distance_slow_2-distance_stop)) * v_max
        else:
             limit_speed = v_max
        
        if 0< limit_speed <0.05:
             limit_speed =0.05

        return limit_speed

    def main(self):
        """
        It spins the wheel
        """
        r = rospy.Rate(30)
        while not rospy.is_shutdown():
            self.caculator_speed_car()
            r.sleep()

if __name__ == '__main__':
    rospy.init_node('Manual_moving')

    safety =  Safety_()
    safety.main()
    rospy.spin()

