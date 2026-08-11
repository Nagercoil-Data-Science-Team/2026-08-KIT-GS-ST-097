import tensorflow as tf
from sklearn.model_selection import train_test_split
# -------------------------------------------------
# Dataset Path
# -------------------------------------------------
train_dir = "archive/images/train"

# -------------------------------------------------
# Parameters
# -------------------------------------------------
IMG_SIZE = (96, 96)  # Increased image size for better pre-trained feature extraction
BATCH_SIZE = 16
SEED = 42

# -------------------------------------------------
# Load Entire Dataset
# -------------------------------------------------
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

# -------------------------------------------------
# Emotion Classes
# -------------------------------------------------
class_names = dataset.class_names

print("\nEmotion Classes")
print(class_names)

# -------------------------------------------------
# Label Encoding
# -------------------------------------------------
label_dict = {name: idx for idx, name in enumerate(class_names)}

print("\nLabel Encoding")
for name, idx in label_dict.items():
    print(f"{name} --> {idx}")

# -------------------------------------------------
# Pixel Normalization
# -------------------------------------------------
normalization_layer = tf.keras.layers.Rescaling(1./255)

dataset = dataset.map(lambda x, y: (normalization_layer(x), y))

print("\nImage Size :", IMG_SIZE)
print("Pixel Normalization : Completed")

# -------------------------------------------------
# Data Augmentation
# -------------------------------------------------
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomZoom(0.20),
    tf.keras.layers.RandomTranslation(
        height_factor=0.10,
        width_factor=0.10
    )
])

print("Data Augmentation :")
print("  Rotation")
print("  Horizontal Flip")
print("  Zoom")
print("  Width & Height Shift")

# -------------------------------------------------
# Dataset Splitting
# -------------------------------------------------
dataset_size = tf.data.experimental.cardinality(dataset).numpy()

train_size = int(0.70 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size

train_dataset = dataset.take(train_size)

remaining = dataset.skip(train_size)

validation_dataset = remaining.take(val_size)

test_dataset = remaining.skip(val_size)

# -------------------------------------------------
# Apply Augmentation only to Training Dataset
# -------------------------------------------------
train_dataset = train_dataset.map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=tf.data.AUTOTUNE
)

train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# -------------------------------------------------
# Command Window Output
# -------------------------------------------------
print("\n==============================")
print("Dataset Splitting")
print("==============================")

print(f"Total Batches      : {dataset_size}")
print(f"Training Batches   : {train_size}")
print(f"Validation Batches : {val_size}")
print(f"Testing Batches    : {test_size}")

print("\nTraining : 70%")
print("Validation : 15%")
print("Testing : 15%")

print("\nPreprocessing Completed Successfully.")

# =================================================
# Step 4: Feature Extraction Using Pre-trained CNN (MobileNetV2)
# =================================================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras import layers, models

num_classes = len(class_names) if class_names else 7

inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

# Our dataset is scaled to [0, 1] by the Rescaling layer.
# MobileNetV2 expects [-1, 1], so we scale it here.
x = (inputs * 2.0) - 1.0

# Pre-trained MobileNetV2 (Transfer Learning)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights='imagenet'
)
# Fine-tune the top layers of the base model
base_model.trainable = True
for layer in base_model.layers[:-30]:  # Freeze the bottom layers
    layer.trainable = False

x = base_model(x)
x = layers.Reshape((-1, x.shape[-1]))(x)

# =================================================
# Step 5: Emotion Classification Using BiLSTM
# =================================================
# Bidirectional Learning & Long Short-Term Memory Networks
x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(64))(x)
x = layers.Dropout(0.4)(x)

# Softmax Classification Layer
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Advanced Learning Rate Schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=1000,
    decay_rate=0.9
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("\nModel Summary (MobileNetV2 + BiLSTM):")
model.summary()

# =================================================
# Training the Model
# =================================================
epochs = 100

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

early_stopping = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

print(f"\nTraining the model for {epochs} epochs...")
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    callbacks=[early_stopping, reduce_lr]
)

# Extract metrics from history for plotting
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# =================================================
# Step 6: Model Evaluation
# =================================================
print("\nEvaluating model on test dataset...")
y_true = []
y_pred_probs = []

for images, labels in test_dataset:
    y_true.extend(labels.numpy())
    preds = model.predict(images, verbose=0)
    y_pred_probs.extend(preds)

y_true = np.array(y_true)
y_pred = np.argmax(y_pred_probs, axis=1)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
cm = confusion_matrix(y_true, y_pred)

print(f"\n==============================")
print("Model Evaluation Metrics")
print("==============================")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")

# =================================================
# Plotting
# =================================================
# Styling settings as requested
plt.rcParams.update({
    'font.size': 18,
    'font.family': 'Times New Roman',
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold'
})

# 1. Accuracy Plot (Separate Window, Separate Colors, No Grid)
fig1 = plt.figure(figsize=(10, 8))
plt.plot(range(1, epochs+1), train_acc, label='Training Accuracy', color='blue', linewidth=3)
plt.plot(range(1, epochs+1), val_acc, label='Validation Accuracy', color='green', linewidth=3)
plt.title('Model Accuracy (MobileNetV2 + BiLSTM)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(False)
fig1.canvas.manager.set_window_title('Accuracy Plot')
plt.show(block=False)

# 2. Loss Plot (Separate Window, Separate Colors, No Grid)
fig2 = plt.figure(figsize=(10, 8))
plt.plot(range(1, epochs+1), train_loss, label='Training Loss', color='red', linewidth=3)
plt.plot(range(1, epochs+1), val_loss, label='Validation Loss', color='orange', linewidth=3)
plt.title('Model Loss (MobileNetV2 + BiLSTM)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(False)
fig2.canvas.manager.set_window_title('Loss Plot')
plt.show(block=False)

# 3. Confusion Matrix Plot (Separate Window, No Grid)
fig3 = plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm',
            xticklabels=class_names if class_names else "auto",
            yticklabels=class_names if class_names else "auto",
            annot_kws={"weight": "bold", "size": 18}, cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(fontweight='bold', rotation=45)
plt.yticks(fontweight='bold', rotation=0)
plt.grid(False)
fig3.canvas.manager.set_window_title('Confusion Matrix')

# Show all plots (blocking)
plt.show()
