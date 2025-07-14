import os
import cv2
import numpy as np
from sklearn.utils import shuffle, class_weight
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# ---------- Load Digits ----------
def load_digit_images(dataset_path='dataset/digits'):
    images, labels = [], []
    for digit in range(10):
        folder_path = os.path.join(dataset_path, str(digit))
        if not os.path.exists(folder_path):
            continue
        for filename in os.listdir(folder_path):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # 0 - 255
                img = cv2.resize(img, (28, 28))
                img = img / 255.0 # 0 - 1.0
                images.append(img)
                labels.append(digit)
    return np.array(images), np.array(labels)

# ---------- Load Operators ----------
operator_label_map = {'plus': 10, 'minus': 11, 'multiply': 12, 'divide': 13}

def load_operator_images(dataset_path='dataset/operators'):
    images, labels = [], []
    for operator, label in operator_label_map.items():
        folder_path = os.path.join(dataset_path, operator)
        if not os.path.exists(folder_path):
            continue
        for filename in os.listdir(folder_path):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (28, 28))
                img = img / 255.0
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)

# ---------- Prepare Dataset ----------
digit_images, digit_labels = load_digit_images()
operator_images, operator_labels_array = load_operator_images()

# Reshape for CNN input
digit_images = digit_images.reshape(-1, 28, 28, 1)
operator_images = operator_images.reshape(-1, 28, 28, 1)

# Combine digits and operators
x_combined = np.concatenate((digit_images, operator_images), axis=0)
y_combined = np.concatenate((digit_labels, operator_labels_array), axis=0)

# Convert labels to categorical one-hot encoding
y_combined_categorical = to_categorical(y_combined, num_classes=14)

# Shuffle dataset
x_combined, y_combined_categorical = shuffle(x_combined, y_combined_categorical, random_state=42)

# Calculate class weights to balance training
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_combined), y=y_combined)
class_weight_dict = dict(enumerate(class_weights))

# ---------- Manual Train/Validation Split ----------
# Stratify by labels to keep class distribution balanced in both sets
x_train, x_val, y_train, y_val = train_test_split(
    x_combined, y_combined_categorical, test_size=0.2, random_state=42, stratify=y_combined
)

# ---------- Data Augmentation ----------
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest'
)

# Fit generator on training data
datagen.fit(x_train)

train_gen = datagen.flow(x_train, y_train, batch_size=32)

# Validation generator without augmentation (just feed data as-is)
val_gen = ImageDataGenerator().flow(x_val, y_val, batch_size=32)

# ---------- Build Model ----------
model = Sequential([
    Input(shape=(28, 28, 1)),  # Use Input layer to fix warning
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),

    Dense(14, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ---------- Callbacks ----------
checkpoint = ModelCheckpoint('best_equation_solver_model.keras', save_best_only=True, monitor='val_accuracy', mode='max')

# ---------- Train Model ----------
model.fit(
    train_gen,
    epochs=30,
    validation_data=val_gen,
    class_weight=class_weight_dict,
    callbacks=[checkpoint]
)

# ---------- Save Final Model ----------
model.save('equation_solver_model.keras')
print("✅ Model trained and saved successfully.")