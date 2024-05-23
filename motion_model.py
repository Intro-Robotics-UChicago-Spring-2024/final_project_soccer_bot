#!/usr/bin/env python3

import rospy
import os
import time

import rospy, cv2, cv_bridge, numpy
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist, Vector3
import cv2
from PIL import Image as im
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import v2
import numpy as np
import torchvision.models as models


"""


rosrun image_transport republish compressed in:=raspicam_node/image raw out:=camera/rgb/image_raw

"""


class MotionModel(object):
    def __init__(self):
        # Initialize this node
        rospy.init_node("motion_model")

        # set up ROS / OpenCV bridge
        self.bridge = cv_bridge.CvBridge()

        # set up cmd_vel publisher
        self.twist_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        self.r = rospy.Rate(10)
        self.rate = 10

        # subscribe to the robot's RGB camera data stream

        self.image_sub = rospy.Subscriber(
            "camera/rgb/image_raw", Image, self.image_callback
        )

        # initialize model
        print('here-1')
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        print('here0')
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)
        print('here1')
        self.model.load_state_dict(torch.load("soccer_bot_model.pth"))
        print('here2')
        self.model.eval()
        print('here3')

        self.image = None

        rospy.sleep(5)
        print('here4')


    def image_callback(self, msg):
        print('here')
        self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
       
        # process image (change colors, convert to PIL, process)

        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        im_pil = im.fromarray(self.image)
        
        preprocess = transforms.Compose([
            transforms.Resize(256),     #change values
            transforms.CenterCrop(224), #change values
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = preprocess(im_pil)

        self.image = input_tensor

        # pass image to model
        pred_action = self.model(input_tensor.unsqueeze(0))

        # get optimal lin and ang velocities
        opt_lin = pred_action[0][0]
        opt_ang = pred_action[0][1]

        twist = Twist()
        twist.linear.x = opt_lin
        twist.angular.z = opt_ang
        self.twist_pub.publish(twist)

    # def send_twist(self):
    #     if self.image is not None:
            
    #         # pass image to model
    #         pred_action = self.model(self.image)

    #         # get optimal lin and ang velocities
    #         opt_lin = pred_action[0][0]
    #         opt_ang = pred_action[0][1]


    #     for i in range(5):
    #         twist = Twist()
    #         twist.linear.x = opt_lin
    #         twist.angular.z = opt_ang
    #         self.twist_pub.publish(twist)
    #         rospy.Rate(10).sleep()

    #     for i in range(2):
    #         # publish stop message
    #         stop_twist = Twist()
    #         self.twist_pub.publish(stop_twist)

    # def run(self):
    #     while True:
    #         self.send_twist()

if __name__ == "__main__":
    model = MotionModel()
    # model.run()
    rospy.spin()

