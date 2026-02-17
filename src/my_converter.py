import os
import glob
import xml.etree.ElementTree as ET
from PIL import Image

def xml_to_yolo_bbox(bbox, w, h):
    # xmin, ymin, xmax, ymax
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [x_center, y_center, width, height]

def convert_annotations(image_dir, xml_dir, output_label_dir, classes):
    if not os.path.exists(output_label_dir):
        os.makedirs(output_label_dir)
        
    all_image_names = set(os.listdir(image_dir))
        
    xml_files = glob.glob(xml_dir + '/*.xml')
    print(f"Found {len(xml_files)} XML files to convert.")
    
    for xml_file in xml_files:
        base_filename = os.path.basename(xml_file)
        # --- FIX FOR PNG FILES ---
        image_name = os.path.splitext(base_filename)[0] + '.png'
        
        if image_name not in all_image_names:
            print(f"Warning: Image '{image_name}' not found for {base_filename}. Skipping.")
            continue

        image_path = os.path.join(image_dir, image_name)
        with Image.open(image_path) as img:
            w, h = img.size
            
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        yolo_annotations = []

        # --- THIS IS THE FINAL, COMPLETED LOGIC ---
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            # Skip if the class name is not in our list
            if class_name not in classes:
                continue
            
            class_id = classes.index(class_name)
            
            bndbox = obj.find('bndbox')
            
            # Get coordinates from XML
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            
            bbox = [xmin, ymin, xmax, ymax]
            
            # Perform the conversion math
            yolo_box = xml_to_yolo_bbox(bbox, w, h)
            
            # Add the formatted string to our list
            yolo_annotations.append(f"{class_id} {' '.join([str(x) for x in yolo_box])}")
        
        # --- This part now works because the list is no longer empty ---
        if yolo_annotations:
            txt_filename = os.path.splitext(base_filename)[0] + '.txt'
            txt_filepath = os.path.join(output_label_dir, txt_filename)
            with open(txt_filepath, 'w') as f:
                f.write('\n'.join(yolo_annotations))

    print(f"Conversion complete! All files saved in {output_label_dir}")
        
if __name__ == "__main__":
    classes = ['pothole']
    
    image_dir = "D:/Pothole_Detection/Pothole-Dataset/images"
    xml_dir = "D:/Pothole_Detection/Pothole-Dataset/annotations"
    output_label_dir = "D:/Pothole_Detection/Pothole-Dataset/labels"

    convert_annotations(image_dir, xml_dir, output_label_dir, classes)
