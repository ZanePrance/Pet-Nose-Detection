# Pet-Nose-Detection
This project utilizes the powerful YOLOv8 object detection model to identify pet noses in images. The included scripts preprocess a dataset of x, y coordinates into bounding boxes, which are then used to train the YOLOv8 network.

# Overview
This project focuses on the creation of a specialized dataset for training an object detection model to locate pet noses within images. The dataset preparation involves converting x, y coordinate annotations into bounding box formats compatible with the YOLOv8 model.

# Dataset Preparation
The script prepare_dataset.py is used to process raw annotations and split the data into training and validation sets. Each annotation is converted into a normalized bounding box format required by YOLOv8

# Training the Model
To train the pet nose localizer, use the following command, making sure to specify the path to your dataset:
python train.py --dataset_path /path/to/your/dataset
The script will initialize the training process of the model using the provided dataset.

# Testing the Model
After training, you can test the model's performance with:
python test.py --test_image_path /path/to/test/images --annotation_path /path/to/annotations --model_weight_path /path/to/model/weights
This will run the model in inference mode on the specified test images and compare the predictions with the ground truth annotations.

# Results
The testing script computes statistical results of the model's performance, including minimum, mean, maximum distances, and standard deviation between the predicted and actual locations of pet noses.

Minimum Distance: <min_distance>
Mean Distance: <mean_distance>
Max Distance: <max_distance>
Standard Deviation: <std_deviation>

These metrics will help you understand the precision of your model.

# Visualization
You can visualize the results of the model by examining the output images with bounding boxes drawn around detected pet noses. Place example images here to showcase the model's detection capabilities:
![image](https://github.com/ZanePrance/Pet-Nose-Detection/assets/141082203/6edb4b9c-d559-4c20-8e13-cb583e1c12fb)

