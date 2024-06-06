import os
import json
import sys

def collect_classes_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    class_set = set()
    stack = [data]

    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if 'class' in node:
                class_set.add(node['class'])
            if 'children' in node:
                stack.extend(node['children'])
        elif isinstance(node, list):
            stack.extend(node)

    return class_set

def collect_classes_from_directory(directory_path):
    all_classes = set()

    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)
            file_classes = collect_classes_from_json(file_path)
            all_classes.update(file_classes)

    class_mapping = {cls: idx for idx, cls in enumerate(sorted(all_classes))}
    return class_mapping

def save_class_mapping(class_mapping, output_file_path):
    with open(output_file_path, 'w') as file:
        json.dump(class_mapping, file, indent=4)
    print(f"Class mapping saved to {output_file_path}")

# Path to the directory containing JSON files
directory_path = "./"
if len(sys.argv) > 1:
    directory_path = sys.argv[1]

# Collect and map classes
class_mapping = collect_classes_from_directory(directory_path)

# Print the class mapping
print(class_mapping)


output_file_path = "./class_mapping.json"
save_class_mapping(class_mapping, output_file_path)