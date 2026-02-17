import os
import random
import shutil

def split_data(image_dir, label_dir, 
               train_images_dir, valid_images_dir, 
               train_labels_dir, valid_labels_dir, 
               split_ratio):
    """
    This function finds all matching image-label pairs, shuffles them, 
    and moves them into training and validation sets.
    """
    
    # 1. Find all confirmed image-label pairs that exist in the source folders.
    all_pairs = []
    for img_name in os.listdir(image_dir):
        if img_name.endswith('.png'):
            name_part = os.path.splitext(img_name)[0]
            lbl_name = name_part + '.txt'
            src_lbl_path = os.path.join(label_dir, lbl_name)
            
            if os.path.exists(src_lbl_path):
                src_img_path = os.path.join(image_dir, img_name)
                all_pairs.append((src_img_path, src_lbl_path))

    # 2. Shuffle the list of pairs randomly.
    random.shuffle(all_pairs)
    
    # 3. Calculate the split point.
    split_point = int(len(all_pairs) * split_ratio)
    
    # 4. Divide the list into training and validation sets.
    train_pairs = all_pairs[:split_point]
    valid_pairs = all_pairs[split_point:]
    
    print(f"Total confirmed pairs found: {len(all_pairs)}")
    print(f"Assigning {len(train_pairs)} to Training set.")
    print(f"Assigning {len(valid_pairs)} to Validation set.")
    
    # 5. A helper function to move the file pairs.
    def move_file_pairs(pairs, dest_img_dir, dest_lbl_dir):
        for src_img_path, src_lbl_path in pairs:
            try:
                # Construct destination paths
                dest_img_path = os.path.join(dest_img_dir, os.path.basename(src_img_path))
                dest_lbl_path = os.path.join(dest_lbl_dir, os.path.basename(src_lbl_path))
                
                # Move the files
                shutil.move(src_img_path, dest_img_path)
                shutil.move(src_lbl_path, dest_lbl_path)
            except FileNotFoundError:
                # If a file is missing (likely from a previous run), don't crash.
                # Just print a warning and continue.
                print(f"Warning: Could not find source file for pair based on {os.path.basename(src_img_path)}. Skipping.")
                continue

    # 6. Move the training file pairs.
    print("\nMoving training files...")
    move_file_pairs(train_pairs, train_images_dir, train_labels_dir)
    
    # 7. Move the validation file pairs.
    print("Moving validation files...")
    move_file_pairs(valid_pairs, valid_images_dir, valid_labels_dir)
    
    print("\nData splitting complete!")


if __name__ == "__main__":
    # --- Using Absolute Paths with Windows-style separators ---
    
    # The source folders where your files are currently located
    image_dir = "D:\\Pothole_Detection\\Pothole-Dataset\\images"
    label_dir = "D:\\Pothole_Detection\\Pothole-Dataset\\labels"

    # The destination folders where the files will be moved
    train_images_dir = 'D:\\Pothole_Detection\\Pothole-Dataset\\images\\train'
    valid_images_dir = 'D:\\Pothole_Detection\\Pothole-Dataset\\images\\valid'
    train_labels_dir = 'D:\\Pothole_Detection\\Pothole-Dataset\\labels\\train'
    valid_labels_dir = 'D:\\Pothole_Detection\\Pothole-Dataset\\labels\\valid'

    # The percentage of data to be used for training (80%)
    split_ratio = 0.8

    # Call the main function to start the process
    split_data(image_dir, label_dir, 
               train_images_dir, valid_images_dir, 
               train_labels_dir, valid_labels_dir, 
               split_ratio)
