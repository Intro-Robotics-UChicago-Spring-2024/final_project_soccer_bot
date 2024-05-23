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
import warnings

warnings.filterwarnings(
    "ignore"
)  # Suppress any warnings so that only velocities are printed to terminal.


"""
This file controls the robots behavior as it follows the predictions of our behavioral cloning model.
As the robot receives images from its camera feed, those images are passed to the model, which
predicts the linear and angular velocities that the robot should take to push the ball into the goal.
In this class, those velocities are turned into the appropriate Twist messages, and passed to the robot
via the cmd_vel topic. Additionally, two extra safety measures are added to ensure that the robot will not
crash into the wall:
1) We trained another classification model that is be able to take in an image and decide if the image was
   taken near the goal or not (and if so, it will send a cmd_vel message to stop the robot). 
2) (Last resort) - on Ctrl+C, the robot will be sent a Twist message that will make it stop. 
"""


class MotionModel(object):
    def __init__(self):
        # Initialize this node
        rospy.init_node("motion_model")

        # set up ROS / OpenCV bridge
        self.bridge = cv_bridge.CvBridge()

        # set up cmd_vel publisher
        self.twist_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        # Set rospy Rate to 10Hz.
        self.r = rospy.Rate(10)
        self.rate = 10

        # subscribe to the robot's RGB camera data stream
        self.image_sub = rospy.Subscriber(
            "camera/rgb/image_raw", Image, self.image_callback
        )

        # On Ctrl+C, the self.stop method will be called
        rospy.on_shutdown(self.stop)

        # Stores most recent image from camera (converted to tensor form)
        self.most_recent_image = None
        # Boolean to tell if the first image from the camera has been received
        self.most_recent_image_set = False

        # initialize model
        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0", "resnet18", pretrained=True
        )
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)
        # Load state dict from the model we trained with all of our data
        self.model.load_state_dict(torch.load("soccer_bot_model_norm_vel.pth"))
        self.model.eval()

        rospy.sleep(5)

    def image_callback(self, msg):
        self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        file_name = "newest_image.jpg"
        # Save the image to the specified directory/name. It needs to be
        # saved to be read in by our secondary classification model (which helps
        # us tell if the robot has reached the goal or not)
        cv2.imwrite(os.path.join("", file_name), self.image)

        # process image (change colors, convert to PIL, process)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        im_pil = im.fromarray(self.image)

        # Transform image to tensor, change size and normalize to match the format of
        # the training data that we used to train our neural netowrk.
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),  # change values
                transforms.CenterCrop(224),  # change values
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        input_tensor = preprocess(im_pil)
        self.most_recent_image = input_tensor
        self.most_recent_image_set = True

    def un_normalize(self, num, angular=False):
        """
        From our training data, we used MinMax normalization with these values:
            MIN LIN VEL: -0.02
            MAX LIN VEL: 0.06
            MIN ANG VEL: -0.2
            MAX ANG VEL: 0.2
        This function un-does the normalization so that the velocities we put in the 
        Twist method for the robot have the correct scale/magnitude.
        """
        min_linear = -0.02
        max_linear = 0.06
        min_angular = -0.2
        max_angular = 0.2

        if angular:
            # return (((num / 100.00) * (max_angular - min_angular)) + min_angular)
            return ((num / 1.0) * (max_angular - min_angular)) + min_angular

        else:
            return ((num / 1.0) * (max_linear - min_linear)) + min_linear

    def run(self):
        while not rospy.is_shutdown():
            if not self.most_recent_image_set:
                # Wait for a few seconds for image_callback function to be called for
                # the first time.
                rospy.sleep(3)
            else:
                # Feed the most recent image into the model, use it to predict the velocities.
                pred_action = self.model(self.most_recent_image.unsqueeze(0))

                # Backup mechanism to classify if the robot has reached the goal or not, based
                # on it's most recent captured image.
                if compute_image_similarity(
                    "/home/tarachugh/catkin_ws/src/final_project_soccer_bot/newest_image.jpg"
                )[1]:
                    # Stop if the robot has reached the goal
                    print("STOPPING")
                    self.stop()
                    rospy.signal_shutdown("Classifier signaled shutdown")
                else:
                    # get optimal lin and ang velocities as predicted by our behavioral cloning model
                    opt_lin = pred_action[0][0]
                    opt_ang = pred_action[0][1]

                    # opt_lin_unnormalized = self.un_normalize(opt_lin)
                    # opt_ang_unnormalized = self.un_normalize(opt_ang, angular=True)
                    opt_lin_unnormalized = opt_lin
                    opt_ang_unnormalized = opt_ang

                    print(
                        f"optimal linear velocity - unnormalized: {opt_lin_unnormalized}"
                    )
                    print(
                        f"optimal angular velocity - unnormalized: {opt_ang_unnormalized}"
                    )

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
