import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Define paths to the dataset directories
train_dir = '/content/Data/train'
val_dir = '/content/Data/valid'
test_dir = '/content/Data/test'

# Set constants
TARGET_SIZE = (224, 224)
BATCH_SIZE = 64
NUM_CLASSES = 4
EPOCHS = 50  # Adjust as needed

# Setting up the ImageDataGenerators for data loading and augmentation
train_generator = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet.preprocess_input,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=False
)

val_test_generator = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet.preprocess_input
)

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

# Build the VGG16 model
def build_vgg16():
    vgg_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    vgg_model.trainable = False
    model = tf.keras.Sequential([
        vgg_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Build the ResNet50 model
def build_resnet50():
    resnet_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    resnet_model.trainable = False
    model = tf.keras.Sequential([
        resnet_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Build the EfficientNetB0 model
def build_efficientnet():
    effnet_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    effnet_model.trainable = False
    model = tf.keras.Sequential([
        effnet_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Train each model
vgg_model = build_vgg16()
history_vgg = vgg_model.fit(train_data, validation_data=val_data, epochs=EPOCHS, verbose=1)

resnet_model = build_resnet50()
history_resnet = resnet_model.fit(train_data, validation_data=val_data, epochs=EPOCHS, verbose=1)

effnet_model = build_efficientnet()
history_effnet = effnet_model.fit(train_data, validation_data=val_data, epochs=EPOCHS, verbose=1)

# Ensemble predictions by averaging the softmax outputs
def ensemble_predictions(models, test_images):
    predictions = [model.predict(test_images) for model in models]
    avg_predictions = np.mean(predictions, axis=0)  # Averaging softmax outputs
    return np.argmax(avg_predictions, axis=1)

# Preprocess test data
test_images, test_labels = next(test_data)

# List of trained models
models = [vgg_model, resnet_model, effnet_model]

# Get ensemble predictions
ensemble_preds = ensemble_predictions(models, test_images)

# Evaluate ensemble predictions
true_labels = np.argmax(test_labels, axis=1)

# Classification report
class_report = classification_report(true_labels, ensemble_preds, target_names=list(test_data.class_indices.keys()))
print(class_report)

# Confusion matrix
conf_matrix = confusion_matrix(true_labels, ensemble_preds)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', xticklabels=test_data.class_indices, yticklabels=test_data.class_indices)
plt.title('Ensemble Confusion Matrix')
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')
plt.show()
