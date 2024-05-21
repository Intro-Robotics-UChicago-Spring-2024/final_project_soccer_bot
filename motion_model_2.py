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
from end_classifier import compute_image_similarity



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
        self.r = rospy.Rate(10.0)
        rospy.on_shutdown(self.stop)
        self.most_recent_image = None


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
        rospy.sleep(5)
        print('here4')


    def image_callback(self, msg):
        print('here')
        self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        file_name = 'newest_image.jpg'
        # Save the image to the specified directory
        cv2.imwrite(os.path.join("", file_name), self.image)
       
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
        self.most_recent_image = input_tensor

    
    def un_normalize(self, num, angular=False):
        """
        MIN LIN VEL: -0.02
        MAX LIN VEL: 0.06
        MIN ANG VEL: -0.2
        MAX ANG VEL: 0.2
        """
        
        min_linear = -.02
        max_linear = .06
        min_angular = -.2
        max_angular = .2

        if angular:
            # return (((num / 100.00) * (max_angular - min_angular)) + min_angular)
            return (((num / 1.0) * (max_angular - min_angular)) + min_angular)

        else:
            return (((num / 1.0) * (max_linear - min_linear)) + min_linear)




    def run(self):
        # pass image to model
        while not rospy.is_shutdown(): 
            if not self.most_recent_image:
                rospy.sleep(3)
            else:
                pred_action = self.model(self.most_recent_image)
                if compute_image_similarity("newest_image.jpg")[1]:
                    self.stop()
                    rospy.signal_shutdown('Classifier signaled shutdown')

                # get optimal lin and ang velocities
                opt_lin = pred_action[0][0]
                opt_ang = pred_action[0][1]

                opt_lin_unnormalized = self.un_normalize(opt_lin)
                opt_ang_unnormalized = self.un_normalize(opt_ang, angular=True)

                print(f"optimal linear velocity - unnormalized: {opt_lin_unnormalized}")
                print(f"optimal angular velocity - unnormalized: {opt_ang_unnormalized}")
                
                twist = Twist()
                twist.linear.x = opt_lin_unnormalized
                twist.angular.z = opt_ang_unnormalized
                self.twist_pub.publish(twist)
                self.r.sleep()
    
    def stop(self):
        self.twist_pub.publish(Twist(linear=Vector3(0, 0, 0), angular=Vector3(0, 0, 0)))
        self.r.sleep()
        



if __name__ == "__main__":
    model = MotionModel()
    # rospy.spin()
    model.run()

