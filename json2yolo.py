import json

# Input JSON data
data = {
  "bounds": [0, 0, 1440, 2392],
  "children": [
    {"text": "Meme Maker", "bounds": [192, 84, 1440, 217], "class": "android.widget.TextView"},
    {"text": "New Meme", "bounds": [192, 171, 1440, 258], "class": "android.widget.TextView"},
    {"text": "Top Text", "bounds": [17, 294, 1422, 381], "class": "android.widget.TextView"},
    {"text": "", "bounds": [17, 381, 1422, 591], "class": "android.widget.EditText"},
    {"text": "Bottom Text", "bounds": [17, 608, 1422, 695], "class": "android.widget.TextView"},
    {"text": "", "bounds": [17, 695, 1422, 905], "class": "android.widget.EditText"},
    {"text": "Credits", "bounds": [17, 922, 1422, 1009], "class": "android.widget.TextView"},
    {"text": "", "bounds": [17, 1009, 1422, 1149], "class": "android.widget.EditText"},
    {"text": "Font Size / Color", "bounds": [17, 1166, 1422, 1253], "class": "android.widget.TextView"},
    {"bounds": [17, 1253, 1212, 1393], "class": "android.widget.SeekBar"},
    {"text": "24", "bounds": [1229, 1253, 1440, 1340], "class": "android.widget.TextView"},
    {"text": "Light Font", "bounds": [35, 1393, 703, 1533], "class": "android.widget.RadioButton"},
    {"text": "Dark Font", "bounds": [738, 1393, 1406, 1533], "class": "android.widget.RadioButton"},
    {"bounds": [17, 1568, 1422, 2392], "class": "android.widget.ImageView"},
    {"bounds": [0, 84, 175, 259], "class": "android.widget.ImageView"},
    {"iconClass": "menu", "bounds": [1265, 84, 1440, 259], "class": "android.widget.ImageView"}
  ]
}

# Image dimensions
image_width, image_height = data['bounds'][2], data['bounds'][3]

# Class mapping (example, modify as needed)
class_mapping = {
    "android.widget.TextView": 0,
    "android.widget.EditText": 1,
    "android.widget.SeekBar": 2,
    "android.widget.RadioButton": 3,
    "android.widget.ImageView": 4
}

# Function to normalize bounding box coordinates
def normalize_bbox(bounds, img_width, img_height):
    x_min, y_min, x_max, y_max = bounds
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return x_center, y_center, width, height

# Generate YOLO annotations
yolo_annotations = []

for child in data['children']:
    bounds = child['bounds']
    class_name = child['class']
    class_id = class_mapping.get(class_name, -1)
    if class_id == -1:
        continue
    normalized_bbox = normalize_bbox(bounds, image_width, image_height)
    annotation = f"{class_id} " + " ".join(map(str, normalized_bbox))
    yolo_annotations.append(annotation)

# Save to file
with open("annotations.txt", "w") as file:
    for annotation in yolo_annotations:
        file.write(annotation + "\n")

print("YOLO annotations generated and saved to annotations.txt")
