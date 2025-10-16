import os, zipfile, shutil, random, requests, math, json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

print("TensorFlow:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

DATA_URL = "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"
ZIP_PATH = "dataset-resized.zip"
EXTRACT_ROOT = "dataset-resized"         # zip extracts a folder named "dataset-resized"
ORIG_DIR = os.path.join(EXTRACT_ROOT, "dataset-resized")  # contains 6 class folders

if not os.path.exists(ZIP_PATH):
    print("Downloading dataset...")
    r = requests.get(DATA_URL, stream=True)
    r.raise_for_status()
    with open(ZIP_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Download complete.")

if not os.path.exists(ORIG_DIR):
    print("Extracting dataset...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(EXTRACT_ROOT)
    print("Extraction complete.")

# Sanity check classes
classes = sorted([d for d in os.listdir(ORIG_DIR) if os.path.isdir(os.path.join(ORIG_DIR, d))])
print("Detected classes:", classes)

SPLIT_DIR = "trashnet_split_optimized"
TRAIN_R, VAL_R, TEST_R = 0.7, 0.15, 0.15

def stratified_split(src_dir, dst_dir, train_r=0.7, val_r=0.2, test_r=0.1, seed=SEED):
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(dst_dir, split), exist_ok=True)

    rng = random.Random(seed)
    classes_local = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]
    classes_local.sort()

    stats = {}
    for cls in classes_local:
        src_c = os.path.join(src_dir, cls)
        images = [f for f in os.listdir(src_c) if os.path.isfile(os.path.join(src_c, f))]
        rng.shuffle(images)
        n = len(images)
        n_train = int(n * train_r)
        n_val   = int(n * val_r)
        n_test  = n - n_train - n_val

        splits = {
            "train": images[:n_train],
            "val":   images[n_train:n_train+n_val],
            "test":  images[n_train+n_val:]
        }

        for sp, files in splits.items():
            sp_dir = os.path.join(dst_dir, sp, cls)
            os.makedirs(sp_dir, exist_ok=True)
            for fname in files:
                shutil.copy(os.path.join(src_c, fname), os.path.join(sp_dir, fname))

        stats[cls] = {k: len(v) for k, v in splits.items()}

    return stats

stats = stratified_split(ORIG_DIR, SPLIT_DIR, TRAIN_R, VAL_R, TEST_R, SEED)
print("Split sizes per class:", json.dumps(stats, indent=2))

IMG_SIZE = (160, 160)     # modest size for speed/quality balance
BATCH_SIZE = 32

# Advanced augmentation to help generalization
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.12,
    height_shift_range=0.12,
    shear_range=0.12,
    zoom_range=0.25,
    channel_shift_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_dir = os.path.join(SPLIT_DIR, "train")
val_dir   = os.path.join(SPLIT_DIR, "val")
test_dir  = os.path.join(SPLIT_DIR, "test")

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', seed=SEED)

val_gen = val_test_datagen.flow_from_directory(
    val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', seed=SEED)

test_gen = val_test_datagen.flow_from_directory(
    test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)

class_indices = train_gen.class_indices
idx_to_class = {v: k for k, v in class_indices.items()}
print("Class indices:", class_indices)

# Compute class weights (handle class imbalance, e.g., 'trash' is small)
train_counts = Counter(train_gen.classes)
max_count = max(train_counts.values())
class_weight = {cls_idx: max_count / cnt for cls_idx, cnt in train_counts.items()}
print("Class weights:", class_weight)


L2 = 1e-4
DROPOUT = 0.35
NUM_CLASSES = train_gen.num_classes

def conv_block(x, filters, k=3, s=1, p='same'):
    x = layers.Conv2D(filters, k, strides=s, padding=p,
                      kernel_regularizer=regularizers.l2(L2), use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

inp = layers.Input(shape=IMG_SIZE + (3,))

x = conv_block(inp, 32); x = conv_block(x, 32); x = layers.MaxPooling2D(2)(x); x = layers.Dropout(0.15)(x)
x = conv_block(x, 64);   x = conv_block(x, 64); x = layers.MaxPooling2D(2)(x); x = layers.Dropout(0.2)(x)
x = conv_block(x, 128);  x = conv_block(x,128); x = layers.MaxPooling2D(2)(x); x = layers.Dropout(0.25)(x)
x = conv_block(x, 192);  x = conv_block(x,192); x = layers.MaxPooling2D(2)(x); x = layers.Dropout(0.3)(x)

x = layers.Conv2D(256, 1, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(L2))(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(DROPOUT)(x)
x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(L2))(x)
x = layers.Dropout(DROPOUT)(x)
out = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = models.Model(inp, out)
model.summary()

# Optimizer + label smoothing (helps noisy labels)
INIT_LR = 3e-4
model.compile(
    optimizer=optimizers.Adam(learning_rate=INIT_LR),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=['accuracy']
)


ckpt_path = "trashnet_cnn_from_scratch_best.h5"
cbs = [
    callbacks.EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    callbacks.ModelCheckpoint(ckpt_path, monitor='val_accuracy', save_best_only=True, verbose=1)
]


EPOCHS = 35
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=cbs,
    verbose=1
)


def plot_curves(h, key='accuracy'):
    plt.figure(figsize=(6,4))
    plt.plot(h.history[key], label=f"train_{key}")
    plt.plot(h.history[f"val_{key}"], label=f"val_{key}")
    plt.title(key.capitalize())
    plt.xlabel("Epoch"); plt.ylabel(key.capitalize())
    plt.legend(); plt.grid(True); plt.show()

plot_curves(history, 'accuracy')
plot_curves(history, 'loss')


test_loss, test_acc = model.evaluate(test_gen, verbose=0)
print(f"Test Accuracy: {test_acc:.2%}")

y_true = test_gen.classes
y_prob = model.predict(test_gen, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

cm = confusion_matrix(y_true, y_pred)
labels = [idx_to_class[i] for i in range(NUM_CLASSES)]

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
plt.tight_layout(); plt.show()

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=labels, digits=4))


model.save("trashnet_cnn_from_scratch_final.h5")
print("Saved: trashnet_cnn_from_scratch_final.h5")
print("Best checkpoint: ", ckpt_path)