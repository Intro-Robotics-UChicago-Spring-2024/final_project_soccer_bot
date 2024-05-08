#!/usr/bin/env python3

import rospy, cv2, cv_bridge
import numpy as np
import os
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Quaternion, Point, Pose, PoseArray, PoseStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header, String
import tf
from tf import TransformListener
from tf import TransformBroadcaster
from tf.transformations import quaternion_from_euler, euler_from_quaternion

def get_yaw_from_pose(p):
        """ A helper function that takes in a Pose object (geometry_msgs) and returns yaw"""

        yaw = (euler_from_quaternion([
                p.orientation.x,
                p.orientation.y,
                p.orientation.z,
                p.orientation.w])
                [2])

        return yaw

class Data_Collection(object):
    def __init__(self):
        rospy.init_node("soccer")
        # once everything is setup initialized will be set to true
        self.initialized = False

        # set the topic names and frame names
        self.base_frame = "base_footprint"
        self.map_topic = "map"
        self.odom_frame = "odom"
        self.scan_topic = "scan"

        # set up ROS / OpenCV bridge
        self.bridge = cv_bridge.CvBridge()

        # rospy.Subscriber('/cmd_vel', Twist, self.callback)
        # rospy.sleep(1)

        # rospy.Subscriber('camera/rgb/image_raw', Image, self.image_callback)
        # rospy.sleep(1)
        # # subscribe to the lidar scan from the robot
        # rospy.Subscriber(self.scan_topic, LaserScan, self.robot_scan_received)
        # rospy.sleep(1)
        

        self.data = []
        # Create a default twist msg (all values 0).
        lin = Vector3()
        ang = Vector3()
        self.move = Twist(linear=lin,angular=ang)
        self.linear_velocity = 0
        self.angular_velocity = 0

        # enable listening for and broadcasting corodinate transforms
        self.tf_listener = TransformListener()
        self.tf_broadcaster = TransformBroadcaster()
        self.start_time = rospy.Time.now().to_sec()

        self.curr_yaw = None
        self.odom_pose = None
        rospy.on_shutdown(self.save_data)



        #need to change folder every time
        self.save_dir = '/home/tarachugh/catkin_ws/src/final_project_soccer_bot/data_coll_5_8'

        rospy.Subscriber('/cmd_vel', Twist, self.callback)
        rospy.sleep(1)

        rospy.Subscriber('camera/rgb/image_raw', Image, self.image_callback)
        rospy.sleep(1)
        # subscribe to the lidar scan from the robot
        rospy.Subscriber(self.scan_topic, LaserScan, self.robot_scan_received)
        rospy.sleep(1)
        self.initialized = True

        

    def robot_scan_received(self, data): # wait until initialization is complete
        if not(self.initialized):
            return

        # we need to be able to transfrom the laser frame to the base frame
        if not(self.tf_listener.canTransform(self.base_frame, data.header.frame_id, data.header.stamp)):
            return

        # wait for a little bit for the transform to become avaliable (in case the scan arrives
        # a little bit before the odom to base_footprint transform was updated)
        self.tf_listener.waitForTransform(self.base_frame, self.odom_frame, data.header.stamp, rospy.Duration(0.5))
        if not(self.tf_listener.canTransform(self.base_frame, data.header.frame_id, data.header.stamp)):
            return

        # calculate the pose of the laser distance sensor
        p = PoseStamped(
            header=Header(stamp=rospy.Time(0),
                          frame_id=data.header.frame_id))

        self.laser_pose = self.tf_listener.transformPose(self.base_frame, p)

        # determine where the robot thinks it is based on its odometry
        p = PoseStamped(
            header=Header(stamp=data.header.stamp,
                          frame_id=self.base_frame),
            pose=Pose())

        self.odom_pose = self.tf_listener.transformPose(self.odom_frame, p)

        self.curr_yaw = get_yaw_from_pose(self.odom_pose.pose)

        #list of the data ranges that lidar picks up
        range_lst = data.ranges
        #finds the smallest value which represents the distance of the closest object, only takes ranges in front of robot
        back_arr = range_lst[530:610]
        #takes out all zeros representing no reading
        filter_close = [x for x in back_arr if x != 0]
        if filter_close:
            self.closest = min(filter_close)
        else:
            self.closest = float('inf')
        
        print(self.closest)

        if self.closest <= .3:
            self.move.linear.x = 0
            self.move.angular.z = 0
            # Publish msg to cmd_vel.
            self.twist_pub.publish(self.move)

            self.save_data[self.data]
    
    def callback(self, msg):
        # Extract linear and angular velocity
        self.linear_velocity = msg.linear.x
        self.angular_velocity = msg.angular.z

    def image_callback(self, msg):
        if not(self.initialized):
            return
        #accesses the image from the camera
        image = self.bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')

        # Generate a unique filename (optional)
        file_name = 'image_' + str(rospy.Time.now().to_sec()) + '.jpg'
        # Save the image to the specified directory
        cv2.imwrite(os.path.join(self.save_dir, file_name), image)
        self.data.append([rospy.Time.now().to_sec(), self.linear_velocity, self.angular_velocity])


    def save_data(self, matrix = None):
        if not matrix:
            matrix = self.data
        #need to change name every time
        path = "robot_run_data_1.csv"
        np.savetxt(path, matrix, fmt='%s')
        print("Saved to:", path)
        return
    
    def run(self):
        rospy.spin()

if __name__ == "__main__":
    node = Data_Collection()
    node.run()
