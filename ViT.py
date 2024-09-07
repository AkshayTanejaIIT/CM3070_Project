import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from transformers import ViTFeatureExtractor, TFViTForImageClassification, create_optimizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Define paths to the dataset directories
train_dir = '/content/Data/train'
val_dir = '/content/Data/valid'
test_dir = '/content/Data/test'

# Setting up constants
TARGET_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_CLASSES = 4
EPOCHS = 50
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100
TOTAL_STEPS = EPOCHS * (train_data.samples // BATCH_SIZE)

# Set up image data generators for augmentation and preprocessing
train_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=False
)

val_test_generator = ImageDataGenerator(rescale=1./255)

# Creating data generators
train_data = train_generator.flow_from_directory(
    train_dir,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_data = val_test_generator.flow_from_directory(
    val_dir,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_data = val_test_generator.flow_from_directory(
    test_dir,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Load the ViT model from Hugging Face with pretrained weights
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
vit_model = TFViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=NUM_CLASSES)

# Freeze some layers if needed
for layer in vit_model.layers[:-2]:  # Freezing all but the last 2 layers
    layer.trainable = False

# Create the Hugging Face optimizer with learning rate scheduling and weight decay
optimizer, schedule = create_optimizer(
    init_lr=LEARNING_RATE,
    num_train_steps=TOTAL_STEPS,
    num_warmup_steps=WARMUP_STEPS,
    weight_decay_rate=WEIGHT_DECAY
)

# Compile the ViT model using the Hugging Face optimizer
vit_model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Define callbacks for early stopping and saving the best model
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(filepath='vit_best_model.keras', save_best_only=True),
    tf.keras.callbacks.TensorBoard(log_dir='./vit_logs')
]

# Preprocessing function for ViT
def preprocess_images(generator):
    images = []
    labels = []
    for batch, label in generator:
        for img in batch:
            img = feature_extractor(images=img, return_tensors='tf').pixel_values[0]
            images.append(img)
        labels.extend(label)
        if len(images) >= generator.samples:
            break
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

# Preprocess the data using ViT's feature extractor
train_images, train_labels = preprocess_images(train_data)
val_images, val_labels = preprocess_images(val_data)
test_images, test_labels = preprocess_images(test_data)

# Train the model
history = vit_model.fit(
    train_images,
    train_labels,
    validation_data=(val_images, val_labels),
    epochs=EPOCHS,
    verbose=1
)

# Evaluate the model on the test set
test_loss, test_accuracy = vit_model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

# Generate predictions and analyze results
predictions_prob = vit_model.predict(test_images).logits
predictions = np.argmax(predictions_prob, axis=1)
true_labels = np.argmax(test_labels, axis=1)

class_report = classification_report(true_labels, predictions, target_names=list(test_data.class_indices.keys()))
print(class_report)

# Confusion matrix
conf_matrix = confusion_matrix(true_labels, predictions)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', xticklabels=test_data.class_indices, yticklabels=test_data.class_indices)
plt.title('Confusion Matrix')
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')
plt.show()

# Plot training and validation accuracy
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
