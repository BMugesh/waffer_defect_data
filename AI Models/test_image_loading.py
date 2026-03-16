"""
Quick test to verify image loading from validation_dataset
"""
import os
from pathlib import Path

DATASET_ROOT = r"c:\Mugi\Projects\IISE\AI model\Dataset"
VAL_DIR = os.path.join(DATASET_ROOT, "validation_dataset")

def discover_classes(directory):
    """Automatically discover class names from directory structure."""
    if os.path.exists(directory):
        classes = sorted([d for d in os.listdir(directory) 
                         if os.path.isdir(os.path.join(directory, d))])
        return classes
    return []

CLASS_NAMES = discover_classes(VAL_DIR)
print(f"Found {len(CLASS_NAMES)} classes: {CLASS_NAMES}")

# Count images per class
for class_name in CLASS_NAMES:
    class_dir = Path(VAL_DIR) / class_name
    
    # Get all image files (case-insensitive, multiple extensions)
    image_files = []
    for ext in ['*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.bmp', '*.BMP']:
        image_files.extend(list(class_dir.glob(ext)))
    
    print(f"  {class_name:15s}: {len(image_files)} images")
    if len(image_files) > 0:
        print(f"    Sample: {image_files[0].name}")

print(f"\n✅ Image loading test complete!")
