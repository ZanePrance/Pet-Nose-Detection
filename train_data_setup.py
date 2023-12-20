import os
import shutil
from PIL import Image
from sklearn.model_selection import train_test_split

annotations_path = r'C:\Users\zane5\PycharmProjects\ELEC 475\Lab5\train_noses3.txt'
images_dir = r'C:\Users\zane5\PycharmProjects\ELEC 475\Lab5\images'
output_dir = r'C:\Users\zane5\PycharmProjects\ELEC 475\Lab5\output'
bbox_size = 10

os.makedirs(os.path.join(output_dir, 'images/train'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'images/val'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels/train'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels/val'), exist_ok=True)

with open(annotations_path, 'r') as f:
    lines = f.readlines()

train_lines, val_lines = train_test_split(lines, test_size=0.2, random_state=42)
def process_annotations(annotations, subset):

    for line in annotations:
        img_name, nose_point_str = line.split(',"')
        nose_point_str = nose_point_str.rstrip('"')
        nose_point_str = nose_point_str.replace('"', '')
        x, y = eval(nose_point_str)

        with Image.open(os.path.join(images_dir, img_name)) as img:
            width, height = img.size

            x_center = x / width
            y_center = y / height
            norm_width = bbox_size / width
            norm_height = bbox_size / height

            label_name = img_name.replace('.jpg', '.txt')
            label_path = os.path.join(output_dir, f'labels/{subset}', label_name)
            with open(label_path, 'w') as label_file:
                label_file.write(f'0 {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n')

            shutil.copy(os.path.join(images_dir, img_name), os.path.join(output_dir, f'images/{subset}', img_name))


process_annotations(train_lines, 'train')
process_annotations(val_lines, 'val')

yaml_content = f"""
path: {output_dir}
train: images/train
val: images/val
nc: 1
names: ['nose']
"""

with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as yaml_file:
    yaml_file.write(yaml_content)




