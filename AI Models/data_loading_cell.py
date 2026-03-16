"""
Complete data loading cell for notebook - uses proper folder structure:
- Synthetic datasets (Bridge, LER, Particles, Other): train/val/test splits already exist
- Real Roboflow datasets: validation_dataset/ and reference_dataset/
"""

# Import train_test_split
from sklearn.model_selection import train_test_split

# Custom data loader to handle flexible directory structure
def load_images_from_folder(folder_path, class_idx, img_size=IMG_SIZE):
    """Load all images from a single folder."""
    images = []
    labels = []
    
    folder = Path(folder_path)
    if not folder.exists():
        return images, labels
    
    # Get all image files (case-insensitive)
    image_files = []
    for ext in ['*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.bmp', '*.BMP']:
        image_files.extend(list(folder.glob(ext)))
    
    for img_path in image_files:
        try:
            img = keras.preprocessing.image.load_img(
                str(img_path),
                target_size=img_size,
                color_mode='grayscale'
            )
            img_array = keras.preprocessing.image.img_to_array(img)
            images.append(img_array)
            labels.append(class_idx)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    
    return images, labels

print("\n" + "="*70)
print("LOADING DATASET FROM PROPER FOLDER STRUCTURE")
print("="*70)

# Initialize storage for each split
train_images = []
train_labels = []
val_images = []
val_labels = []
test_images = []
test_labels = []

# Define 8 classes and their data sources
# Format: class_name: {'train': [...], 'val': [...], 'test': [...]}
class_data_sources = {
    'bridge': {
        'train': [os.path.join(DATASET_ROOT, 'Bridge', 'train')],
        'val': [
            os.path.join(DATASET_ROOT, 'Bridge', 'val'),
            os.path.join(DATASET_ROOT, 'validation_dataset', 'bridge')
        ],
        'test': [
            os.path.join(DATASET_ROOT, 'Bridge', 'test'),
            os.path.join(DATASET_ROOT, 'reference_dataset', 'bridge')
        ]
    },
    'clean': {
        'train': [],  # No training data for clean
        'val': [os.path.join(DATASET_ROOT, 'validation_dataset', 'clean')],
        'test': []
    },
    'cmp_scratch': {
        'train': [],
        'val': [os.path.join(DATASET_ROOT, 'validation_dataset', 'cmp_scratch')],
        'test': []
    },
    'crack': {
        'train': [],
        'val': [os.path.join(DATASET_ROOT, 'validation_dataset', 'crack')],
        'test': []
    },
    'film_residue': {
        'train': [],
        'val': [os.path.join(DATASET_ROOT, 'validation_dataset', 'film_residue')],
        'test': []
    },
    'ler': {
        'train': [os.path.join(DATASET_ROOT, 'LER', 'train')],
        'val': [os.path.join(DATASET_ROOT, 'LER', 'val')],
        'test': [os.path.join(DATASET_ROOT, 'LER', 'test')]
    },
    'other': {
        'train': [os.path.join(DATASET_ROOT, 'Other', 'train')],
        'val': [os.path.join(DATASET_ROOT, 'Other', 'val')],
        'test': [os.path.join(DATASET_ROOT, 'Other', 'test')]
    },
    'particles': {
        'train': [os.path.join(DATASET_ROOT, 'Particles', 'train')],
        'val': [
            os.path.join(DATASET_ROOT, 'Particles', 'val'),
            os.path.join(DATASET_ROOT, 'validation_dataset', 'particles')
        ],
        'test': [
            os.path.join(DATASET_ROOT, 'Particles', 'test'),
            os.path.join(DATASET_ROOT, 'reference_dataset', 'particles')
        ]
    }
}

# Load data for each class
for class_idx, class_name in enumerate(CLASS_NAMES):
    print(f"\n{'='*70}")
    print(f"Loading {class_name.upper()}")
    print(f"{'='*70}")
    
    # Load training data
    train_count = 0
    for source_folder in class_data_sources[class_name]['train']:
        images, labels = load_images_from_folder(source_folder, class_idx)
        if len(images) > 0:
            train_images.extend(images)
            train_labels.extend(labels)
            train_count += len(images)
            print(f"  TRAIN - {Path(source_folder).parent.name}/{Path(source_folder).name}: {len(images)} images")
    
    # Load validation data
    val_count = 0
    for source_folder in class_data_sources[class_name]['val']:
        images, labels = load_images_from_folder(source_folder, class_idx)
        if len(images) > 0:
            val_images.extend(images)
            val_labels.extend(labels)
            val_count += len(images)
            print(f"  VAL   - {Path(source_folder).parent.name}/{Path(source_folder).name}: {len(images)} images")
    
    # Load test data
    test_count = 0
    for source_folder in class_data_sources[class_name]['test']:
        images, labels = load_images_from_folder(source_folder, class_idx)
        if len(images) > 0:
            test_images.extend(images)
            test_labels.extend(labels)
            test_count += len(images)
            print(f"  TEST  - {Path(source_folder).parent.name}/{Path(source_folder).name}: {len(images)} images")
    
    print(f"  ✅ Total {class_name}: Train={train_count}, Val={val_count}, Test={test_count}")

# Convert to numpy arrays
X_train = np.array(train_images) / 255.0
y_train = keras.utils.to_categorical(train_labels, num_classes=NUM_CLASSES)

X_val = np.array(val_images) / 255.0
y_val = keras.utils.to_categorical(val_labels, num_classes=NUM_CLASSES)

X_test = np.array(test_images) / 255.0
y_test = keras.utils.to_categorical(test_labels, num_classes=NUM_CLASSES)

# Display summary
print("\n" + "="*70)
print("DATASET SUMMARY")
print("="*70)
print(f"Training:   {X_train.shape[0]} images")
print(f"Validation: {X_val.shape[0]} images")
print(f"Test:       {X_test.shape[0]} images")
print(f"Total:      {X_train.shape[0] + X_val.shape[0] + X_test.shape[0]} images")
print(f"Image shape: {X_train.shape[1:]}")

print(f"\n{'='*70}")
print("CLASS DISTRIBUTION")
print(f"{'='*70}")
print(f"{'Class':<15} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
print("-"*70)

train_counts = Counter(train_labels)
val_counts = Counter(val_labels)
test_counts = Counter(test_labels)

for class_idx, class_name in enumerate(CLASS_NAMES):
    t_count = train_counts.get(class_idx, 0)
    v_count = val_counts.get(class_idx, 0)
    te_count = test_counts.get(class_idx, 0)
    total = t_count + v_count + te_count
    print(f"{class_name:<15} {t_count:>8} {v_count:>8} {te_count:>8} {total:>8}")

print("="*70)
