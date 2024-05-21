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


def compute_image_similarity(new_image_path, threshold=0.87):
    # Load pre-trained model
    model = models.resnet50(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])  # Remove the classification layer
    model.eval()  # Set to evaluation mode

    # Transformation to match the model's expected input
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def extract_features(image_path):
        image = Image.open(image_path).convert('RGB')
        image = preprocess(image)
        image = image.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            features = model(image)
        return features.squeeze().numpy()

    # Function to compute cosine similarity
    def cosine_sim(vec1, vec2):
        return cosine_similarity([vec1], [vec2])[0][0]
    def get_image_files(folder_path):
        valid_extensions = ('.jpg', '.jpeg', '.png') 
        return [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.lower().endswith(valid_extensions)]



    # Get image paths
    image_paths = get_image_files('/Users/tarachugh/Desktop/robotics/final_project_soccer_bot/end_images')

    # Check if the folder contains images
    if not image_paths:
        raise ValueError("No images found in the folder.")

    # Extract features for each image in the folder
    features_list = [extract_features(path) for path in image_paths]

    # Extract features for the new image to compare
    new_image_features = extract_features(new_image_path)

    # Compute similarities
    similarities = [cosine_sim(new_image_features, features) for features in features_list]

    # Calculate mean similarity score
    mean_similarity_score = sum(similarities) / len(similarities)

    # Determine if new image is similar enough
    is_similar = mean_similarity_score > threshold

    return mean_similarity_score, is_similar
