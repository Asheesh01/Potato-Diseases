# ============================================================
#  EVALUATE POTATO DISEASE MODEL ACCURACY
# ============================================================
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# -------------------- CONFIG --------------------
MODEL_PATH = "potato_vgg16.h5"
DATASET_PATH = "PotatoDataset/test"   # ✅ Path to your test folder
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# -------------------- LOAD MODEL --------------------
print("🔄 Loading model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# -------------------- LOAD TEST DATA --------------------
print("📦 Loading test dataset...")
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# -------------------- EVALUATE --------------------
print("🧪 Evaluating model...")
loss, accuracy = model.evaluate(test_generator, verbose=1)
print(f"\n✅ Test Accuracy: {accuracy * 100:.2f}%")
print(f"📉 Test Loss: {loss:.4f}")

# -------------------- PREDICTIONS --------------------
print("🔍 Generating predictions...")
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# -------------------- CONFUSION MATRIX --------------------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

# -------------------- CLASSIFICATION REPORT --------------------
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

# -------------------- ACCURACY PLOT --------------------
try:
    history_path = "training_history.npy"
    if os.path.exists(history_path):
        history = np.load(history_path, allow_pickle=True).item()
        plt.figure(figsize=(8, 4))
        plt.plot(history['accuracy'], label='Train Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.legend()
        plt.title("Training Accuracy Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.show()
    else:
        print("\n⚠️ No saved training history found (skipping curve).")
except Exception as e:
    print(f"⚠️ Could not plot training curve: {e}")

print("\n✅ Evaluation complete.")
