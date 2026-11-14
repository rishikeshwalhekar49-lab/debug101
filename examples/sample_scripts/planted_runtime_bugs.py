
import tensorflow as tf
import numpy as np
import json
import time

# ==========================================
# 1. LOAD MNIST DATASET
# ==========================================

(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print("MNIST Dataset Loaded")
print("Training samples:", len(x_train))
print("Testing samples:", len(x_test))


# ==========================================
# 2. CREATE NEURAL NETWORK
# ==========================================

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

loss_function = tf.keras.losses.SparseCategoricalCrossentropy()

# ==========================================
# 3. TRAINING CONFIGURATION
# ==========================================

EPOCHS = 5
BATCH_SIZE = 32

training_logs = []

# ==========================================
# 4. TRAINING LOOP WITH MONITORING
# ==========================================

for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    epoch_losses = []
    gradient_norms = []
    activation_means = []
    weight_changes = []

    dataset = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)
    ).shuffle(10000).batch(BATCH_SIZE)

    for step, (images, labels) in enumerate(dataset):

        # Save weights before update
        old_weights = [
            w.numpy().copy() for w in model.trainable_weights
        ]

        with tf.GradientTape() as tape:

            predictions = model(images, training=True)

            loss = loss_function(labels, predictions)

        # Calculate gradients
        gradients = tape.gradient(
            loss, model.trainable_variables
        )

        # Calculate gradient norm
        grad_norm = tf.linalg.global_norm(gradients).numpy()

        # Apply gradients
        optimizer.apply_gradients(
            zip(gradients, model.trainable_variables)
        )

        # Calculate weight changes
        changes = []

        for old, new in zip(
            old_weights, model.trainable_weights
        ):
            changes.append(
                float(np.mean(np.abs(new.numpy() - old)))
            )

        weight_change = float(np.mean(changes))

        # Get hidden layer activations
        activation_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[
                model.layers[2].output,
                model.layers[3].output
            ]
        )

        activations = activation_model(
            images, training=False
        )

        activation_mean = float(
            tf.reduce_mean(activations[0]).numpy()
        )

        # Store metrics
        epoch_losses.append(float(loss.numpy()))
        gradient_norms.append(float(grad_norm))
        activation_means.append(activation_mean)
        weight_changes.append(weight_change)

        # Print progress
        if step % 200 == 0:

            print(
                f"Step {step} | "
                f"Loss: {loss.numpy():.4f} | "
                f"Gradient Norm: {grad_norm:.4f} | "
                f"Weight Change: {weight_change:.8f}"
            )

    # ==========================================
    # 5. EVALUATION
    # ==========================================

    test_loss, test_accuracy = model.evaluate(
        x_test, y_test, verbose=0
    )

    # Record epoch metrics
    log = {
        "epoch": epoch + 1,
        "train_loss": float(np.mean(epoch_losses)),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "gradient_norm": float(np.mean(gradient_norms)),
        "activation_mean": float(np.mean(activation_means)),
        "weight_change": float(np.mean(weight_changes))
    }

    training_logs.append(log)

    print("\nEpoch Summary:")
    print(json.dumps(log, indent=4))


# ==========================================
# 6. SAVE TRAINING LOGS
# ==========================================

with open("mnist_training_logs.json", "w") as file:

    json.dump(training_logs, file, indent=4)


# ==========================================
# 7. SAVE MODEL
# ==========================================

model.save("mnist_digit_model.keras")

print("\nTraining completed!")
print("Training logs saved!")
print("Model saved!")


# ==========================================
# 8. FINAL EVALUATION
# ==========================================

print("\nFINAL RESULTS")

print("Test Accuracy:", test_accuracy * 100, "%")
print("Test Loss:", test_loss)
