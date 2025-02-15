# Project Description: 

The goal of our project (“Soccer Bot”) was to have a robot learn to push a ball into a goal via an optimal path, as learned from expert demonstrations. Soccer bot uses a behavioral cloning approach to determine the optimal action at each state. At each timestamp, the robot passes the image from its camera feed into a model that predicts the angular and linear velocity needed to navigate towards the goal, and these velocities are passed to the robot via a Twist message on the cmd_vel topic. We were able to train a model that successfully navigated the robot and ball to the goal in linear paths from multiple starting points/angles, pushed the ball into the goal, and stopped a safe distance away from the goal. Notably, our implementation does not use any LiDAR data, teleoping, or real-time image processing to determine where it is or what action it needs to take; it simply passes images from its camera into our model, and takes action accordingly. Our main goals were (1) to learn how to implement a behavioral cloning algorithm, (2) process camera and cmd_vel data to train a neural network, and (3) be able to show that our model successfully led to the robot being able to push the ball into the goal. 

We thought that a behavioral cloning algorithm would be interesting to learn about and implement, because of the nature of its reward system. Behavioral cloning is a type of imitation learning algorithm, which is a type of reinforcement learning where the explicit reward function is not known but is learned through “expert demonstrations.” In the past projects from this class (like the q-learning project), the reward function was known and could be explicitly calculated when the algorithm was deciding which next step would be optimal, but this is not possible in a behavioral cloning algorithm. In our case, the expert demonstrations were us teleop-ing the robot to push the ball into the goal from many different starting points. Behavioral supervised learning to determine a policy Π that imitates the expert, and the goal is to minimize the difference between the learned policy and the expert demonstration. It essentially solves this optimization problem:
<img width="839" alt="Screenshot 2024-05-23 at 7 53 24 PM" src="https://github.com/Intro-Robotics-UChicago-Spring-2024/final_project_soccer_bot/assets/118474712/2b03e139-0e40-4964-8086-1c41993affe3">


The main components of our project were (1) data collection, (2) training our neural network model, and (3) determining the optimal action at each step (using our model and the physical robot). 

Before we could start collecting data, we had to build an attachment for the robot that pushes the ball ahead of the robot, so that the camera can see the ball while it pushes it. After iterating on many prototypes, we were able to make an attachment out of a paper towel roll, an index card, two markers, and two rulers, which could successfully hold and push the ball, and allow the ball to be far enough in front of the camera to be seen by the camera. We also tried out multiple types of balls to find one whose dynamics met the needs of our system, and we settled on a wiffle ball. The wiffle ball is heavy enough to not roll away while the robot is pushing it, and it is brightly colored so the camera can more easily identify it.


Testing different types of balls:

![dodgeball](https://github.com/Intro-Robotics-UChicago-Spring-2024/final_project_soccer_bot/assets/118474712/4c5117ff-e4a7-47f6-81a8-f4afceaa87c9)

![ezgif-6-cb15979024](https://github.com/Intro-Robotics-UChicago-Spring-2024/final_project_soccer_bot/assets/118474712/0d4a32be-521d-476b-bcab-2bd917d7d3cd)


Wiffle ball:

![wiffle](https://github.com/Intro-Robotics-UChicago-Spring-2024/final_project_soccer_bot/assets/118474712/01c7934c-7f6d-4a5a-b224-dc9aa2982684)

Attachment arm:

<img width="371" alt="Screenshot 2024-05-22 at 12 22 40 AM" src="https://github.com/Intro-Robotics-UChicago-Spring-2024/final_project_soccer_bot/assets/118474712/06036535-76db-4b53-8e76-bd2d458729ec">




For the data collection part, we collected data from “expert” runs teleoping the robot to push the ball into the goal. At each timestamp, we stored the image from the robot’s camera and its simultaneous angular and linear velocities. We did this for 25 runs, and collected around 200 data points from each run (pictures and corresponding velocities); these runs all had different starting points and trajectories toward the goal. 

Then, using the data collected as our “expert” runs, we trained the network with the robot’s camera feed and cmd_vel messages at different time steps. We used PyTorch’s ResNet-18 (18-layers deep CNN) and trained it with 80% of our data. In order to train this network, we first had to process the data by normalizing the images and velocities. Then, we initialized the model and the Mean Squared Error loss function, trained the model, then tested/validated the model using the 20% of data we had set aside for testing. We trained two different models, one that gives data using classification and one that gives data based on the error that the predicted value has from the target. For each of these model types, we trained many models with different parameters (like different input datasets and numbers of epochs) so that we could choose the most successful one.

Lastly, we created a motion model, which sets up an image callback function which carries out the preferred behavior at each timestep. At each tilmestep, we pass the image to the trained neural net, which returns values for the optimal angular and linear velocities, then we publish a corresponding twist message. We noticed that our model was not able to consistently stop the robot when it got close enough to the goal (see the ‘Challenges’ section for more on this), so as a back up strategy to try to stop the robot at the goal, we decided to use another classification model that would be able to take in an image and decide if the image was taken at the goal or not (and if so, it would send a cmd_vel message to stop the robot). This helped ensure that the robot did not crash into the wall, and also allowed us to stick to one of the goals of our project, which was to use only camera data to decide the robots actions.



# System Architecture:

## 1. Data Collection

To collect data from our "expert" runs, we had used the file data_collection.py. We set up a ROS node and subscribed to the camera feed and the cmd_vel topic. In the image callback function, we saved the image received to a folder with the time stamp as the file name. In the cmd_vel callback function, we recorded the linear velocity, angular velocity, and time stamp, which we saved to a csv at the end. We collected data for 25 runs with different starting positions and taking different paths towards the goal by teleop-ing the robot. The data for each run was saved into a separate folder. 

The code for data collection is located in data_collection.py. In the Data_Collection class, subscribers are initialized to record the turtlebot's LiDAR, cmd_vel, and image. An array is used to store the data at each timestep that the robot's state is received for, with both the image and velocties recorded (in image_callback, while callback is used to update the class variables storing the current velocities). Position is extracted in the robot_scan_received() function, calculating as a Pose and used to indicate the final part of a data collection run, when an object in front of the turtlebot is sensed within the range of 0.3m, to indicate the goal/a wall has been reached. At this point, the velocities can be instead saved as 0 to indicate in the data that the robot must be stopped. Finally, save_data() uses numpy's savetxt() method to save the array in the current dierectory.


## 2. Neural Network

The code for the neural network is in create_model.py, while create_classification_model.py contains an alternate build for creating the model through classifiying the outputs into bins rather than a gradient of values.

Structure is largely the same across these 2 solutions, with the main() function running everything needed to load the data, initialize the model, and save it.

The supporting class of Data_Process reads in the data from the data collection, taking in a directory of multiple runs, and for each run, saving each of the images after they are processed into a tensor in an array. For the time, velocity, and quality of the runs, the csv is converted into a dataframe and each row has the velocities added in chronological order to another array.

These two arrays are loaded into SoccerBotDataset class to convert them into a pytorch dataset.

For the model's initialization, values are specified for the number of epochs and loss function, with the loss function using either the pytorch neural network's mean squared error or their cross entropy loss function (depending on if the classification model was used). In the actual initialization of the model, the constructor for the MyModel() class is called. The specific model resnet 18 is used, and two additional layers are added to the neural network for the velocities. 

The train() function takes our model and trains it each epoch, noting the difference between the predicted and actual values are with the loss function, backwards propogates the gradient and then updates the paramaters using the gradient, which is done by imported pytorch functions backward() and optimizer.step()

The test() function works similarly to train() except the gradient generated is not used to update the model, and instead only notes how well the model performs.

The train() functions gets performed on 80% of the data, and the test() function on the other 20%.

Finally, the model is saved after all epochs have been ran.


## 3. Motion Model

The code for this is in motion_model_2.py. This file has a MotionModel class, which sets up a rospy node, with a subscriber to the camera/rgb/image_raw topic and a publisher to the cmd_vel topic. It also initializes our behavioral cloning model (from the neural network that we described above). 

The stop_bot method is used to send a stopping Twist message to the robot (in the case of a backup ‘emergency stop’ when the model does not predict the robot should stop, and it is about to crash into the wall). The un_normalize method is used to convert the output of the model prediction back to its un-normalized velocities which are the correct magnitude/scale to be passed to the robot in a Twist message.

In the image_callback method, the image is converted to a cv2 image, which is then saved to a file called newest_img.jpg. It must be saved to a file in order for the code in end_classifier.py to work correctly. Then,  the image is processed - its colors are converted to RGB, and the image is transformed to a  tensor, with a new size and it is normalized. This is necessary for the image to match the format of the data that we used to train our neural network, and to match the expected input to our model. 

The ‘run’ method has the main loop that is run once every cycle (the node is set up to run at a rate of 10Hz, so the loop runs every 1/10 second). This method first checks if the image callback function has ever been called, and if not it sleeps for a few seconds to wait for the image data to start being populated. Once photos are being received from the camera, this method passes the most recent photograph into the model to predict the linear and angular velocities that the robot should follow at this timestep. It puts those in a Twist object and publishes that to the cmd_vel topic. The ‘run’ method also uses a backup strategy to ensure that the robot does indeed stop when it gets close enough to the goal. Using the saved image (from the image_callback function), it passes that image to the compute_image_similarity function from end_classifier.py, and if the image is close enough to the set of images from the expert runs where the robot has reached the goal, the robot is stopped.

End_classifier.py includes a function that takes in a path to an image taken on the robot's camera, and it compares it to a dataset of images that we put together (the last image from each of our expert runs). If the image is similar enough to one in the dataset, we determine that the robot has reached the goal (and should stop). This is a backup strategy in case our main model does not make the robot stop at the goal, and we implemented this so that the robot would not crash into the wall. In this function, first the PyTorch reset 50 (50-layer CNN model) is loaded in, in a pre-trained state. The classification layer is removed and the model is set to evaluation mode, as we are just using this to extract features from images. For all of the files in the end_images folder (which includes the last image from each expert run, where the robot is at the goal), features are extracted using the resnet50 model. This feature extraction happens through first preprocessing the image (changing its size, type, and normalizing it to match the expected inputs to the resnet50 model) Then, features are extracted from the image that we are trying to classify. For each image in the end_images folder, the cosine similarity is computed between that image and the image we are trying to classify. We used the cosine_similarity function from the sklearn.metrics.pairwise library to do this. Cosine similarity is a way to measure how similar two vectors are. It is calculated as the dot product of the vectors over the product of their magnitudes. Here, the features of the images are stored in vectors, so we can compute the cosine similarity of them like we would with purely numerical vectors. Here’s a useful graphic from https://www.learndatasci.com/glossary/cosine-similarity/ that describes cosine similarity: 

![cosine-similarity-vectors original](https://github.com/Intro-Robotics-UChicago-Spring-2024/final_project_soccer_bot/assets/118474712/34b1822b-ede4-4bf9-89bd-cb8826ca6c5d)

Then, the mean similarity score is calculated (averaging out the similarity scores between the image to classify and all of the images in the end_images set). Through trial and error, we determined a threshold similarity score of 87, and if the score is higher we determine that the robot is at the goal (and thus should stop). The compute_image_similarity function in end_classifier.py returns both the mean similarity score and a boolean of if similarity passes the threshold. 


# ROS Node Diagram 

<img width="755" alt="Screenshot 2024-05-23 at 7 38 54 PM" src="https://github.com/Intro-Robotics-UChicago-Spring-2024/final_project_soccer_bot/assets/114627557/c9197b08-ff05-4fe6-a1fd-10adfc70cbc4">



# Execution: 
In order to train our models (only needs to be done once):
- Non-Classifier Model (predicts velocities on a continuous scale):
    - Run create_model.py (i.e. with “python3 create_model.py”), and make sure to change self.data_dir (which is now set to “self.data_dir = '/home/gnakanishi/catkin_ws/src/final_project_soccer_bot/test_data’” to the directory that holds the data for each run, where each run’s folder includes all of the pictures for that run and the csv of velocities for that run)
- Classifier Model (bins velocities):
    - Run create_classification_model.py (i.e. with “python3 create_classification_model.py”), and make sure to change self.data_dir (which is now set to “self.data_dir = '/home/gnakanishi/catkin_ws/src/final_project_soccer_bot/test_data’” to the directory that holds the data for each run, where each run’s folder includes all of the pictures for that run and the csv of velocities for that run)


In order to run the motion model (test the code with the ball/goal setup):
- Terminal 1: roscore
- Terminal 2: ssh into the robot’s raspberry pi, set_ip to your machine’s address, bringup
- Terminal 3: ssh into the robot’s raspberry pi, set_ip to your machine’s address, bringup_cam
- Terminal 4: rosrun image_transport republish compressed in:=raspicam_node/image raw out:=camera/rgb/image_raw
- Terminal 5: rosrun final_project_soccer_bot motion_model_2.py
    - In the line that reads “if compute_image_similarity("/home/tarachugh/catkin_ws/src/final_project_soccer_bot/newest_image.jpg”)[1]:”, you will need to change the file path to be the path to your final_project_soccer_bot directory (motion_model_2.py will create a file called newest_image.jpg there)

# Challenges, Future Work, and Takeaways: 
###These should take a similar form and structure to how you approached these in the previous projects (1 paragraph each for the challenges and future work and a few bullet points for takeaways)

## Challenges
One of the problems we faced in training our behavioral cloning models (neural networks) was that initially, the outputs and predictions we were seeing were not at all in line with the data we took, especially in terms of the ratio between the linear and angular velocities at each timestep. In the data we took, most of the ‘expert’ runs were straight-line paths to the goal, with linear velocities in the -.02 m/s to 0.6 m/s range and angular velocities in the -0.2 radians/s to 0.2 radians/s range. The vast majority of the time, the magnitude of the angular velocity was well below that of the linear velocity. However, we found that this was not reflected in our model’s outputs. After looking more closely at the data that we were feeding into the model and consulting the TAs for help, we found that we would need to normalize our input data. This would scale both the linear and angular velocities to be between 0 and 1, which would put them on ‘equal footing’ when entering the model, and it would mean that the loss function would factor them with equal importance. This change helped us see more physically meaningful predictions from our model. Another challenge that we faced was that all of our models (we tried classification/binning models as well as models that predicted velocities on a continuous scale) seemed to have trouble correctly predicting that the robot was meant to stop near the goal. We hypothesize that this is because when we were collecting data from expert runs and teleoping the robot, we stopped data collection before or immediately after stopping the robot (we were more focused on getting data for the robot’s path to the goal, not for the robot stopping at the goal), and thus there were only a few data points with images corresponding to 0 linear and angular velocities. As a back up mechanism to try to stop the robot at the goal, we decided to use another classification model that would be able to take in an image and decide if the image was taken at the goal or not (and if so, it would send a cmd_vel message to stop the robot). We used a pretrained PyTorch ResNet50 model to extract the features from the last image that the robot captured in each expert run (where the robot would be at the goal), then when we inputted an image it would compute the cosine similarity of the inputted image to the set of training images, and if the images were similar enough, it would stop the robot. This added an extra layer of protection to our implementation, so that the robot would not hit the wall, and still allowed us to use image-related models (rather than lidar data or teleop-ing) to do this.

## Future Work
One goal for the future would be to make the robot find the ball or goal when it is not in view, and then push the ball toward the goal. In all of the successful runs of our model and implementation, the robot had to have the goal and ball in view in order to successfully score the goal. However, we took data for a few expert runs where the ball and goal were not in view of the robot’s camera and the robot rotated to find them and then scored the goal. We found that our trained model did capture some of this behavior - when we put the robot down without the ball and goal in view, it did indeed start rotating in place. However, we were not able to observe a successful run with our model where the robot rotated to see the ball, navigated to the ball, and scored a goal. This is likely due to fact that we only had ~2 expert runs where the robot did this, so perhaps collecting more data and training our model on that would be useful to achieve this. Here is a picture of what the robot might see at the beginning of one of these runs:

<img width="281" alt="Screenshot 2024-05-21 at 10 51 43 PM" src="https://github.com/Intro-Robotics-UChicago-Spring-2024/final_project_soccer_bot/assets/118474712/51a0a607-8619-4d22-bfd5-729446b89809">



 If we had more time, it would also be useful to have the robot be able to maneuver around obstacles on its way toward the goal. When we were testing our model, we noticed that often people walked between the robot and goal while the robot was driving toward the goal, and it would be useful for our robot to be able to react accordingly. We would need to collect more expert run data with obstacles and train our model based on that data, in order for the robot to be able to learn this behavior. It would probably be more reasonable to start with trying to navigate around static obstacles, compared to moving obstacles like people walking by. Here is a picture of an obstacle captured by the robot that it would need to navigate around:
 
<img width="289" alt="Screenshot 2024-05-21 at 10 51 49 PM" src="https://github.com/Intro-Robotics-UChicago-Spring-2024/final_project_soccer_bot/assets/118474712/4baf90e4-14af-4876-852c-4dd105fbe8b1">


Lastly, a future goal of our project could be to have the robot learn from and imitate more complicated (swerving) trajectories. We collected data from swerving trajectories and trained our model on it, but it seems that the model outputs tend to have the robot always move in a straight line to the goal. In order to have the model successfully learn to move in swerving trajectories, perhaps we would need to reward the model differently for predicting these trajectories (trajectories with larger angular velocities) in order to encourage the model to output such trajectories.

## Takeaways
- We learned about the importance of the training data in a machine learning algorithm. As we saw in our demo, there were cases that our algorithm was not well trained on, like when the ball was not in front of the robot or when the ball was removed from the arm. We now know that we need to train on more diverse data for this type of algorithm to handle edge cases.
- We also learned about working on a large project with a group. When we were working on different components, we had to catch up on other group members' progress in order to work on the next parts. We also had to manage the changes everyone was making to the code and make sure we were merging it together well.
- It was also beneficial to develop the project, deliverables, and implementation plan on our own. Having to create the project instead of just following given directions made us learn how to accomplish a project with more independence and responsibility.









# DEMO:

Here are two example runs of the robot scoring a goal. The robot starts and ends in slightly different positions in the two runs, and successfully stops at the goal in both runs (without using any LiDAR information or teleoping!). You can also notice that the terminal window on the laptop in the videos is populating with the linear and angular speeds that are predicted by our model (and sent as cmd_vel messages to the robot) at each timestep when the robot receives a picture. We have also included a picture of what that terminal output looks like. Note that the videos are sped up to 2x speed.



https://github.com/Intro-Robotics-UChicago-Spring-2024/final_project_soccer_bot/assets/118474712/00024596-ae9c-44c4-8903-9501fd9663e9


https://github.com/Intro-Robotics-UChicago-Spring-2024/final_project_soccer_bot/assets/118474712/48f1c063-6134-422c-ae2a-564185af663f

<img width="543" alt="Output" src="https://github.com/Intro-Robotics-UChicago-Spring-2024/final_project_soccer_bot/assets/118474712/7d5a1e78-7173-4f86-89ef-5ca7d00dea31">
