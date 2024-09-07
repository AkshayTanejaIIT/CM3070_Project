import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam

def build_model(hparams):
    """Builds an EfficientNetB0 model based on hyperparameters."""
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    base_model.trainable = True  # Fine-tune all layers

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(hparams['dropout_rate']),
        Dense(4, activation='softmax')  # Assuming 4 classes
    ])

    model.compile(
        optimizer=Adam(learning_rate=hparams['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Hyperparameters grid
learning_rates = [1e-5, 1e-4, 1e-3]
batch_sizes = [16, 32]
dropout_rates = [0.3, 0.5]

# Dictionary to store the results
results = {}

for lr in learning_rates:
    for batch_size in batch_sizes:
        for dropout in dropout_rates:
            hparams = {
                'learning_rate': lr,
                'dropout_rate': dropout
            }
            model = build_model(hparams)
            print(f"Training with LR={lr}, Batch Size={batch_size}, Dropout={dropout}")
            history = model.fit(
                train_data,
                validation_data=val_data,
                epochs=10,  # You might adjust this for full training
                batch_size=batch_size,
                verbose=1
            )
            
            # Evaluate the model on the validation set
            val_accuracy = model.evaluate(val_data)[1]  # Get accuracy score
            results[(lr, batch_size, dropout)] = val_accuracy
            print(f"Completed LR={lr}, Batch Size={batch_size}, Dropout={dropout} -> Val Accuracy: {val_accuracy}")

# Identify the best performing hyperparameters
best_params = max(results, key=results.get)
best_accuracy = results[best_params]
print(f"Best Parameters: LR={best_params[0]}, Batch Size={best_params[1]}, Dropout={best_params[2]}")
print(f"Best Validation Accuracy: {best_accuracy}")
