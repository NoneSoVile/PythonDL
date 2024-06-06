import json
import numpy as np

np.set_printoptions(suppress=True)

# Function to normalize bounding box coordinates
def normalize_bbox(bounds, img_width, img_height):
    x_min, y_min, x_max, y_max = bounds
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return x_center, y_center, width, height

# Class mapping (example, modify as needed)
class_mapping = {
    "android.widget.TextView": 0,
    "android.widget.EditText": 1,
    "android.widget.SeekBar": 2,
    "android.widget.RadioButton": 3,
    "android.widget.ImageView": 4,
    "menu": 5  # Example class for iconClass
}

# Function to load simplified JSON file and convert to YOLO format
def convert_json_to_yolo(json_file_path, yolo_output_path):
    with open(json_file_path, "r") as file:
        data = json.load(file)

    # Image dimensions
    image_width, image_height = data['bounds'][2], data['bounds'][3]

    # Generate YOLO annotations
    yolo_annotations = []

    for child in data['children']:
        bounds = child['bounds']
        class_name = child.get('class', '')
        if not class_name and 'iconClass' in child:
            class_name = child['iconClass']
        class_id = class_mapping.get(class_name, -1)
        if class_id == -1:
            continue
        normalized_bbox = normalize_bbox(bounds, image_width, image_height)
        annotation = f"{class_id} " + " ".join(f"{coord:.5f}" for coord in normalized_bbox)
        yolo_annotations.append(annotation)

    # Save YOLO annotations to file
    with open(yolo_output_path, "w") as file:
        for annotation in yolo_annotations:
            file.write(annotation + "\n")

    print(f"YOLO annotations generated and saved to {yolo_output_path}")

# Convert the saved JSON file to YOLO format
convert_json_to_yolo("simplified_data.json", "annotations.txt")
