import json
import os
import shutil
from pathlib import Path
import sys
import yaml
def convert_bbox_to_yolo(size, bbox):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = bbox[0] + bbox[2] / 2.0
    y = bbox[1] + bbox[3] / 2.0
    w = bbox[2]
    h = bbox[3]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_segmentation_to_yolo(size, segmentation):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x_points = segmentation[0][::2]  # Extract x-coordinates
    y_points = segmentation[0][1::2]  # Extract y-coordinates
    x_min, x_max = min(x_points), max(x_points)
    y_min, y_max = min(y_points), max(y_points)
    x = (x_min + x_max) / 2.0
    y = (y_min + y_max) / 2.0
    w = x_max - x_min
    h = y_max - y_min
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_coco_to_yolo(coco_json_path, images_dir, output_dir):
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    images = {img['id']: img for img in coco_data['images']}
    
    output_image_dir = os.path.join(output_dir, 'images')
    output_label_dir = os.path.join(output_dir, 'labels')

    Path(output_image_dir).mkdir(parents=True, exist_ok=True)
    Path(output_label_dir).mkdir(parents=True, exist_ok=True)

    for ann in coco_data['annotations']:
        yolo_segmentation = None
        image_id = ann['image_id']
        category_id = ann['category_id']
        bbox = ann['bbox']
        segmentation = ann['segmentation']

        image_info = images[image_id]
        image_file_name = image_info['file_name']
        image_width = image_info['width']
        image_height = image_info['height']
        
        yolo_bbox = convert_bbox_to_yolo([image_width, image_height], bbox)
        
        if segmentation is not None and len(segmentation) > 0:
            yolo_bbox = convert_segmentation_to_yolo((image_width, image_height), segmentation)
        
        label_file_name = os.path.splitext(image_file_name)[0] + '.txt'
        label_file_path = os.path.join(output_label_dir, label_file_name)
        
        with open(label_file_path, 'a') as label_file:           
            if yolo_segmentation is not None:
                label_file.write(f"{category_id - 1} {' '.join(map(str, yolo_bbox))} \n")
            else:
                label_file.write(f"{category_id - 1} {' '.join(map(str, yolo_bbox))} \n")
        
        # Copy image to output directory
        shutil.copy(os.path.join(images_dir, image_file_name), os.path.join(output_image_dir, image_file_name))
    return categories

def process_dataset(input_dir, output_dir):
    # Remove existing output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    class_names = None
    for split in ['train', 'valid']:
        split_input_dir = os.path.join(input_dir, split)
        split_output_dir = os.path.join(output_dir, split)
        
        coco_json_path = os.path.join(split_input_dir, '_annotations.coco.json')
        images_dir = os.path.join(split_input_dir, '')
        
        categories = convert_coco_to_yolo(coco_json_path, images_dir, split_output_dir)
        if class_names is None:
            class_names = [categories[i] for i in sorted(categories.keys())]
    
    class_names.pop(0)        
    print("len of class_names", len(class_names))
    # Create data.yaml
    data_yaml = {
        'train': '../train/images',
        'val': '../valid/images',
        'nc': len(class_names),
        'names': str(class_names)
    }
    
    # Format the names string to match the desired style
    strconvert = json.dumps(class_names)
    strconvert = strconvert.replace("\"", "'")
    print(strconvert)
    data_yaml['names'] = str(strconvert)
    
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

if __name__ == "__main__":
    input_dir = "E:\\testdatasets\\objdetection2.v3i.coco"
    output_dir = "E:\\testdatasets\\yolocoverted"
    
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
        
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    if input_dir == "":
        print("please input the diretory of coco dataset")
        exit()
        
    if output_dir == "":
        print("please  enter the diretory of converted yolo dataset")
        exit()
    
    process_dataset(input_dir, output_dir)
