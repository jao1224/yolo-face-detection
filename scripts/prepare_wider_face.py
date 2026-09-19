import os
from datasets import load_dataset
from PIL import Image

# Load the dataset
print("Loading dataset...")
dataset = load_dataset("CUHK-CSE/wider_face")

# Define the splits we want to use
splits = ['train', 'val']  # We'll use the official train and val splits

# Base directory for saving
base_dir = 'train'
images_dir = os.path.join(base_dir, 'images')
labels_dir = os.path.join(base_dir, 'labels')

# Create directories
for split in splits:
    os.makedirs(os.path.join(images_dir, split), exist_ok=True)
    os.makedirs(os.path.join(labels_dir, split), exist_ok=True)

# Process each split
for split in splits:
    print(f"Processing {split} split...")
    for i, example in enumerate(dataset[split]):
        # Get image and annotations
        image = example['image']
        # The image is already a PIL Image object
        # Get bounding boxes: list of [x, y, width, height] in pixels
        bboxes = example['bbox']
        # Get image dimensions
        width, height = image.size

        # Save image
        image_filename = f"{i:06d}.jpg"
        image_path = os.path.join(images_dir, split, image_filename)
        image.save(image_path)

        # Save labels in YOLO format
        label_filename = f"{i:06d}.txt"
        label_path = os.path.join(labels_dir, split, label_filename)
        with open(label_path, 'w') as f:
            for bbox in bboxes:
                x, y, w, h = bbox
                # Convert to YOLO format: x_center, y_center, width, height (normalized)
                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height
                w_norm = w / width
                h_norm = h / height
                # Class index for face is 0
                f.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        if (i+1) % 1000 == 0:
            print(f"  Processed {i+1} images")

print("Done!")