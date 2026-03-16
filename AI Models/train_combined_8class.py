"""
Semiconductor Defect Detection - 8 Classes (Combined Dataset)
Combines real Roboflow images + synthetic images for comprehensive training

Real classes (validation_dataset): bridge, clean, cmp_scratch, crack, film_residue, particles
Synthetic classes (Dataset): Bridge, LER, Particles, Other

Final 8 classes: bridge, clean, cmp_scratch, crack, film_residue, LER, Other, particles
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
from collections import Counter
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# GPU configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU Available: {len(gpus)} device(s)")
    except RuntimeError as e:
        print(e)
else:
    print("⚠️  No GPU detected, using CPU")

print(f"✅ TensorFlow version: {tf.__version__}")

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET_ROOT = r"c:\Mugi\Projects\IISE\AI model\Dataset"
OUTPUT_DIR = r"c:\Mugi\Projects\IISE\AI model\outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define all 8 classes (standardized to lowercase)
CLASS_NAMES = ['bridge', 'clean', 'cmp_scratch', 'crack', 'film_residue', 'ler', 'other', 'particles']
NUM_CLASSES = len(CLASS_NAMES)

# Model configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Training configuration
INITIAL_EPOCHS = 5
FINE_TUNE_EPOCHS = 20
INITIAL_LR = 0.001
FINE_TUNE_LR = 0.0001

print("="*70)
print("CONFIGURATION - COMBINED DATASET (REAL + SYNTHETIC)")
print("="*70)
print(f"Dataset root: {DATASET_ROOT}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"\nImage size: {IMG_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"\nClasses ({NUM_CLASSES}):")
for i, cls in enumerate(CLASS_NAMES, 1):
    print(f"  {i}. {cls}")
print(f"\nInitial LR: {INITIAL_LR}")
print(f"Fine-tune LR: {FINE_TUNE_LR}")
print("="*70)

# ============================================================================
# DATA LOADING FROM MULTIPLE SOURCES
# ============================================================================

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
print("LOADING COMBINED DATASET")
print("="*70)

all_images = []
all_labels = []

# Mapping of class names to their sources
class_sources = {
    'bridge': [
        os.path.join(DATASET_ROOT, 'validation_dataset', 'bridge'),  # Real images
        os.path.join(DATASET_ROOT, 'Bridge', 'train'),  # Synthetic train
        os.path.join(DATASET_ROOT, 'Bridge', 'val'),    # Synthetic val
        os.path.join(DATASET_ROOT, 'Bridge', 'test'),   # Synthetic test
    ],
    'clean': [
        os.path.join(DATASET_ROOT, 'validation_dataset', 'clean'),
    ],
    'cmp_scratch': [
        os.path.join(DATASET_ROOT, 'validation_dataset', 'cmp_scratch'),
    ],
    'crack': [
        os.path.join(DATASET_ROOT, 'validation_dataset', 'crack'),
    ],
    'film_residue': [
        os.path.join(DATASET_ROOT, 'validation_dataset', 'film_residue'),
    ],
    'ler': [
        os.path.join(DATASET_ROOT, 'LER', 'train'),
        os.path.join(DATASET_ROOT, 'LER', 'val'),
        os.path.join(DATASET_ROOT, 'LER', 'test'),
    ],
    'other': [
        os.path.join(DATASET_ROOT, 'Other', 'train'),
        os.path.join(DATASET_ROOT, 'Other', 'val'),
        os.path.join(DATASET_ROOT, 'Other', 'test'),
    ],
    'particles': [
        os.path.join(DATASET_ROOT, 'validation_dataset', 'particles'),  # Real images
        os.path.join(DATASET_ROOT, 'Particles', 'train'),  # Synthetic train
        os.path.join(DATASET_ROOT, 'Particles', 'val'),    # Synthetic val
        os.path.join(DATASET_ROOT, 'Particles', 'test'),   # Synthetic test
    ],
}

# Load images from all sources
for class_idx, class_name in enumerate(CLASS_NAMES):
    class_image_count = 0
    print(f"\nLoading {class_name}:")
    
    for source_folder in class_sources[class_name]:
        images, labels = load_images_from_folder(source_folder, class_idx)
        if len(images) > 0:
            all_images.extend(images)
            all_labels.extend(labels)
            class_image_count += len(images)
            print(f"  {Path(source_folder).parent.name}/{Path(source_folder).name}: {len(images)} images")
    
    print(f"  Total for {class_name}: {class_image_count} images")

# Convert to numpy arrays
X_all = np.array(all_images) / 255.0
y_all = keras.utils.to_categorical(all_labels, num_classes=NUM_CLASSES)

print("\n" + "="*70)
print("DATASET LOADED")
print("="*70)
print(f"Total images: {len(X_all)}")
print(f"\nClass distribution:")
class_counts = Counter(all_labels)
for class_idx in sorted(class_counts.keys()):
    print(f"  {CLASS_NAMES[class_idx]:15s}: {class_counts[class_idx]:4d} images")
print("="*70)

# ============================================================================
# SPLIT DATA
# ============================================================================

print("\n" + "="*70)
print("SPLITTING DATA (70/15/15)")
print("="*70)

# First split: 70% train, 30% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X_all, y_all, test_size=0.3, random_state=42, stratify=np.argmax(y_all, axis=1)
)

# Second split: 15% val, 15% test (50/50 of the 30%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=np.argmax(y_temp, axis=1)
)

print(f"Training:   {X_train.shape[0]} images")
print(f"Validation: {X_val.shape[0]} images")
print(f"Test:       {X_test.shape[0]} images")
print(f"Total:      {X_all.shape[0]} images")
print(f"Image shape: {X_train.shape[1:]}")

print(f"\nClass distribution (training):")
train_class_counts = Counter(np.argmax(y_train, axis=1))
for class_idx in sorted(train_class_counts.keys()):
    print(f"  {CLASS_NAMES[class_idx]:15s}: {train_class_counts[class_idx]:4d} images")
print("="*70)

# ============================================================================
# VISUALIZE SAMPLES
# ============================================================================

samples_per_class = 4
fig, axes = plt.subplots(NUM_CLASSES, samples_per_class, figsize=(12, 3*NUM_CLASSES))
fig.suptitle(f'Sample Images from Each Category ({NUM_CLASSES} classes)', 
             fontsize=16, fontweight='bold')

for class_idx, class_name in enumerate(CLASS_NAMES):
    class_indices = np.where(np.argmax(y_train, axis=1) == class_idx)[0]
    
    if len(class_indices) == 0:
        continue
    
    sample_count = min(samples_per_class, len(class_indices))
    sample_indices = np.random.choice(class_indices, sample_count, replace=False)
    
    for i, idx in enumerate(sample_indices):
        ax = axes[class_idx, i]
        ax.imshow(X_train[idx].squeeze(), cmap='gray')
        ax.axis('off')
        if i == 0:
            ax.set_title(class_name, fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'sample_images_combined_8class.png'), dpi=150, bbox_inches='tight')
plt.show()
print("✅ Sample visualization saved")

# ============================================================================
# BUILD MODEL
# ============================================================================

def build_model(num_classes=NUM_CLASSES, img_size=IMG_SIZE):
    """Build MobileNetV3-Small model for grayscale defect classification."""
    
    inputs = layers.Input(shape=(*img_size, 1), name='input_layer')
    
    # Convert grayscale to RGB
    x = layers.Concatenate()([inputs, inputs, inputs])
    
    # Load MobileNetV3-Small
    base_model = MobileNetV3Small(
        input_shape=(*img_size, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    
    base_model.trainable = False
    
    x = base_model(x, training=False)
    
    # Classification head
    x = layers.Dropout(0.3, name='dropout_1')(x)
    x = layers.Dense(256, activation='relu', name='dense_1')(x)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.Dropout(0.2, name='dropout_2')(x)
    x = layers.Dense(128, activation='relu', name='dense_2')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    
    outputs = layers.Dense(num_classes, activation='softmax', name='output_layer')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name=f'DefectDetector_{num_classes}Class')
    
    return model, base_model

model, base_model = build_model()

print("\n" + "="*70)
print("MODEL ARCHITECTURE")
print("="*70)
print(f"Total parameters: {model.count_params():,}")
print(f"Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
print(f"Non-trainable parameters: {sum([tf.size(w).numpy() for w in model.non_trainable_weights]):,}")
print("="*70)

# ============================================================================
# PHASE 1: TRAIN CLASSIFIER HEAD
# ============================================================================

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=INITIAL_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

callbacks_phase1 = [
    ModelCheckpoint(
        os.path.join(OUTPUT_DIR, 'phase1_best_combined.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
]

print("\n" + "="*70)
print("PHASE 1: TRAINING CLASSIFIER HEAD")
print("="*70)
print(f"Epochs: {INITIAL_EPOCHS}")
print(f"Learning rate: {INITIAL_LR}")
print(f"Base model frozen: {not base_model.trainable}")
print("="*70 + "\n")

history_phase1 = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=INITIAL_EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks_phase1,
    verbose=1
)

print("\n✅ Phase 1 training complete!")

# ============================================================================
# PHASE 2: FINE-TUNE ENTIRE MODEL
# ============================================================================

base_model.trainable = True

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

callbacks_phase2 = [
    ModelCheckpoint(
        os.path.join(OUTPUT_DIR, 'best_model_combined_8class.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    TensorBoard(
        log_dir=os.path.join(OUTPUT_DIR, 'logs'),
        histogram_freq=1
    )
]

print("\n" + "="*70)
print("PHASE 2: FINE-TUNING ENTIRE MODEL")
print("="*70)
print(f"Epochs: {FINE_TUNE_EPOCHS}")
print(f"Learning rate: {FINE_TUNE_LR}")
print(f"Base model frozen: {not base_model.trainable}")
print(f"Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
print("="*70 + "\n")

history_phase2 = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=FINE_TUNE_EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks_phase2,
    verbose=1
)

print("\n✅ Phase 2 fine-tuning complete!")

model.save(os.path.join(OUTPUT_DIR, 'final_model_combined_8class.h5'))
print(f"✅ Final model saved")

# ============================================================================
# TRAINING HISTORY VISUALIZATION
# ============================================================================

def plot_training_history(history1, history2, output_dir):
    combined_history = {
        'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
        'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
        'loss': history1.history['loss'] + history2.history['loss'],
        'val_loss': history1.history['val_loss'] + history2.history['val_loss']
    }
    
    epochs_phase1 = len(history1.history['accuracy'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(combined_history['accuracy'], label='Training Accuracy', linewidth=2)
    ax1.plot(combined_history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax1.axvline(x=epochs_phase1-0.5, color='red', linestyle='--', label='Fine-tuning starts', alpha=0.7)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(combined_history['loss'], label='Training Loss', linewidth=2)
    ax2.plot(combined_history['val_loss'], label='Validation Loss', linewidth=2)
    ax2.axvline(x=epochs_phase1-0.5, color='red', linestyle='--', label='Fine-tuning starts', alpha=0.7)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history_combined_8class.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 Final Training Accuracy: {combined_history['accuracy'][-1]:.4f}")
    print(f"📊 Final Validation Accuracy: {combined_history['val_accuracy'][-1]:.4f}")
    print(f"📊 Best Validation Accuracy: {max(combined_history['val_accuracy']):.4f}")

plot_training_history(history_phase1, history_phase2, OUTPUT_DIR)
print("✅ Training history visualization saved")

# ============================================================================
# MODEL EVALUATION
# ============================================================================

best_model = keras.models.load_model(os.path.join(OUTPUT_DIR, 'best_model_combined_8class.h5'))

print("\n" + "="*70)
print("EVALUATING ON TEST SET")
print("="*70)

test_loss, test_accuracy, test_precision, test_recall = best_model.evaluate(
    X_test, y_test,
    batch_size=BATCH_SIZE,
    verbose=1
)

print("\n" + "="*70)
print("TEST SET RESULTS")
print("="*70)
print(f"Test Loss:      {test_loss:.4f}")
print(f"Test Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Test Precision: {test_precision:.4f} ({test_precision*100:.2f}%)")
print(f"Test Recall:    {test_recall:.4f} ({test_recall*100:.2f}%)")
print(f"Test F1-Score:  {2 * (test_precision * test_recall) / (test_precision + test_recall):.4f}")
print("="*70)

# ============================================================================
# CONFUSION MATRIX
# ============================================================================

y_pred_probs = best_model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred)

def plot_enhanced_confusion_matrix(cm, class_names, output_dir):
    """Plot beautiful, annotated confusion matrix."""
    
    fig_height = max(10, NUM_CLASSES * 1.2)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    annotations = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            percent = cm_percent[i, j]
            annotations[i, j] = f"{count}\n({percent:.1f}%)"
    
    sns.heatmap(
        cm,
        annot=annotations,
        fmt='',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Number of Predictions'},
        linewidths=1,
        linecolor='gray',
        ax=ax
    )
    
    ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
    ax.set_title(f'Confusion Matrix - {NUM_CLASSES}-Class Combined Dataset', 
                 fontsize=15, fontweight='bold', pad=20)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_combined_8class.png'), dpi=200, bbox_inches='tight')
    plt.show()
    
    print("✅ Enhanced confusion matrix saved")

plot_enhanced_confusion_matrix(cm, CLASS_NAMES, OUTPUT_DIR)

# ============================================================================
# CLASSIFICATION REPORT
# ============================================================================

print("\n" + "="*70)
print("DETAILED CLASSIFICATION REPORT")
print("="*70)
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))
print("="*70)

print("\nPER-CLASS ACCURACY:")
for i, class_name in enumerate(CLASS_NAMES):
    class_mask = (y_true == i)
    if sum(class_mask) > 0:
        class_acc = accuracy_score(y_true[class_mask], y_pred[class_mask])
        print(f"  {class_name:15s}: {class_acc:.4f} ({class_acc*100:.2f}%)")

report_dict = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
with open(os.path.join(OUTPUT_DIR, 'classification_report_combined_8class.json'), 'w') as f:
    json.dump(report_dict, f, indent=2)

print("\n✅ Classification report saved")

# ============================================================================
# EXPORT TO TFLITE
# ============================================================================

converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

tflite_path = os.path.join(OUTPUT_DIR, 'defect_detector_combined_8class.tflite')
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

h5_size = os.path.getsize(os.path.join(OUTPUT_DIR, 'best_model_combined_8class.h5')) / (1024 * 1024)
tflite_size = os.path.getsize(tflite_path) / (1024 * 1024)

print("\n" + "="*70)
print("MODEL EXPORT SUMMARY")
print("="*70)
print(f"Keras model (.h5):     {h5_size:.2f} MB")
print(f"TFLite model (.tflite): {tflite_size:.2f} MB")
print(f"Size reduction:         {(1 - tflite_size/h5_size)*100:.1f}%")
print(f"\nTFLite model saved to: {tflite_path}")
print("="*70)

print("\n✅ Model exported for edge deployment!")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

summary = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'dataset': {
        'total_images': X_all.shape[0],
        'train': X_train.shape[0],
        'val': X_val.shape[0],
        'test': X_test.shape[0],
        'classes': CLASS_NAMES,
        'num_classes': NUM_CLASSES,
        'sources': 'Combined: Roboflow real images + Synthetic generated images'
    },
    'model': {
        'architecture': 'MobileNetV3-Small',
        'total_parameters': int(model.count_params()),
        'input_shape': list(IMG_SIZE) + [1],
        'output_classes': NUM_CLASSES
    },
    'training': {
        'phase1_epochs': INITIAL_EPOCHS,
        'phase2_epochs': FINE_TUNE_EPOCHS,
        'initial_lr': INITIAL_LR,
        'fine_tune_lr': FINE_TUNE_LR,
        'batch_size': BATCH_SIZE
    },
    'performance': {
        'test_accuracy': float(test_accuracy),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'test_f1': float(2 * (test_precision * test_recall) / (test_precision + test_recall))
    },
    'model_files': {
        'keras_model': 'best_model_combined_8class.h5',
        'tflite_model': 'defect_detector_combined_8class.tflite',
        'keras_size_mb': float(h5_size),
        'tflite_size_mb': float(tflite_size)
    }
}

with open(os.path.join(OUTPUT_DIR, 'training_summary_combined_8class.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*70)
print(f"TRAINING COMPLETE - {NUM_CLASSES}-CLASS COMBINED DATASET")
print("="*70)
print(f"\n📊 Dataset: {summary['dataset']['total_images']} images ({NUM_CLASSES} classes)")
print(f"   Sources: Real Roboflow + Synthetic Generated")
print(f"🏗️  Model: {summary['model']['architecture']} ({summary['model']['total_parameters']:,} parameters)")
print(f"\n🎯 Test Performance:")
print(f"   Accuracy:  {summary['performance']['test_accuracy']:.4f} ({summary['performance']['test_accuracy']*100:.2f}%)")
print(f"   Precision: {summary['performance']['test_precision']:.4f} ({summary['performance']['test_precision']*100:.2f}%)")
print(f"   Recall:    {summary['performance']['test_recall']:.4f} ({summary['performance']['test_recall']*100:.2f}%)")
print(f"   F1-Score:  {summary['performance']['test_f1']:.4f} ({summary['performance']['test_f1']*100:.2f}%)")
print(f"\n💾 Model Files:")
print(f"   Keras (.h5):     {summary['model_files']['keras_size_mb']:.2f} MB")
print(f"   TFLite (.tflite): {summary['model_files']['tflite_size_mb']:.2f} MB")
print(f"\n📁 Output directory: {OUTPUT_DIR}")
print("="*70)

print("\n✅ All files saved successfully!")
print("\n🚀 Model ready for edge deployment!")
