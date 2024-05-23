import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

"""
This function takes in a path to an image taken on the robot's camera, and it
compares it to a dataset of images that we put together (the last image from each
of our expert runs). If the image is similar enough to one in the dataset, we determine
that the robot has reached the goal (and should stop). This is a backup strategy in case our
main model does not make the robot stop at the goal, and we implemented this so that the robot would 
not crash into the wall.
"""


def compute_image_similarity(
    new_image_path, threshold=0.87
):  # Threshold of 0.87 gotten by trial and error.
    model = models.resnet50(pretrained=True)  # Load Pytorch 50-layer CNN model
    model = nn.Sequential(
        *list(model.children())[:-1]
    )  # Remove classification layer as we won't need to use that
    model.eval()  # Set model to evaluation mode

    # Change size and normalize image from robot to match standardized inputs for resnet50 model
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def extract_features(image_path):
        image = Image.open(image_path).convert("RGB")  # Open image, change colors
        image = preprocess(image)  # preprocess image to be put into the pytorch model
        image = image.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            features = model(image)  # Extract features from the image, convert to numpy
        return features.squeeze().numpy()

    # Compute cosine similarity between two vectors, which is described here:
    # https://onyekaokonji.medium.com/cosine-similarity-measuring-similarity-between-multiple-images-f289aaf40c2b.
    # Cosine similarity tells us how similar the two images are, and we use the cosine similarity function
    # that is built into the scikit-learn library.
    def cosine_sim(vec1, vec2):
        return cosine_similarity([vec1], [vec2])[0][0]

    # Gets all image files out of a given directory (this is used for the expert run images).
    def get_image_files(folder_path):
        valid_extensions = (".jpg", ".jpeg", ".png")
        return [
            os.path.join(folder_path, file)
            for file in os.listdir(folder_path)
            if file.lower().endswith(valid_extensions)
        ]

    # Get image paths for each last image from the expert runs.
    image_paths = get_image_files("end_images")

    # Check if the folder contains images
    if not image_paths:
        raise ValueError("Folder has no images.")

    # Extract features for each image using the pytorch resnet50 model.
    features_list = [extract_features(path) for path in image_paths]

    # Extract features for new image using the pytorch resnet50 model.
    new_image_features = extract_features(new_image_path)

    # Compute cosine similarity between new image and each image in the expert images dataset.
    similarities = [
        cosine_sim(new_image_features, features) for features in features_list
    ]

    # Calculate mean similarity score.
    mean_similarity_score = sum(similarities) / len(similarities)

    # Determine if new image is similar enough
    is_similar = mean_similarity_score > threshold

    # Return similarity score and a boolean of if similarity passes the threshold.
    return mean_similarity_score, is_similar
