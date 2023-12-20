import torch
import numpy as np
import argparse
from ultralytics import YOLO
from testDataloader import TestLoader

parser = argparse.ArgumentParser(description='Test the pet nose localizer.')
parser.add_argument('-testIm', '--test_image_path', type=str, help='Path to test images')
parser.add_argument('-', '--annotation_path', type=str, help='Path to image annotations')
parser.add_argument('-mw', '--model_weight_path', type=str, help='Path to model weight')

# Parse the arguments
args = parser.parse_args()

# Use the arguments
test_image_path = args.test_image_path
annotation_path = args.annotation_path
model_weight = args.model_weight_path

#load the trained model

model = YOLO(model_weight)

def eucliden_distance(pred, true):
    return ((pred[0] - true[0]) ** 2 + (pred[1] - true[1]) ** 2) ** 0.5



# set up the test dataloader
test_loader = TestLoader(test_image_path, annotation_path)

distances = []
print('Running in inference mode...')
for img, truth in test_loader:

    with torch.no_grad():
        results = model(img)

    if len(results) > 0:
        result = results[0]

        if len(result.boxes) > 0:
            pred_box = result.boxes[0]
            flattened_xywhn = pred_box.xywhn.flatten()

            if flattened_xywhn.numel() >= 4:
                # extract the normalized x and y center
                pred_x, pred_y = flattened_xywhn[0].item(), flattened_xywhn[1].item()

                distance = eucliden_distance((pred_x, pred_y), truth)
                distances.append(distance)
            else:
                print("unexpected box format")
                print(pred_box.xywhn)


# compute statistics
min_distance = np.min(distances)
mean_distance = np.mean(distances)
max_distance = np.max(distances)
std_deviation = np.std(distances)

print(f'Minimum Distance: {min_distance}')
print(f'Mean Distance: {mean_distance}')
print(f'Max Distance: {max_distance}')
print(f'Standard Deviation: {std_deviation}')


