"""This module is for test purposes to create a simple CNN 
module that will be converted to OpenVino for use on NPU.
It is following the tensorflow tutorial 
https://www.tensorflow.org/tutorials/images/cnn"""

import logging
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import openvino as ov
import tf2onnx
# import onnx
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logging.basicConfig(filename='test.log',level=logging.INFO)

MODEL_PATH = Path("models")


def get_data():
    """Get Cifar10 dataset and return train-test split data
    """
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    return train_images, test_images, train_labels, test_labels

def plot_images(images, labels):

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])
        # The CIFAR labels happen to be arrays, 
        # which is why you need the extra index
        plt.xlabel(class_names[labels[i][0]])
    plt.show()

def build_model(api="sequential"):  
    logging.info("Build model...")

    if api == "functional":

        # Input layer
        inputs = layers.Input(shape=(32, 32, 3))

        # Convolutional layers
        x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)

        # Flatten and Dense layers
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(10)(x)

        # Define the model
        model = models.Model(inputs=inputs, outputs=outputs)
    else:
        model = models.Sequential()
        model.add(layers.Input(shape=(32, 32, 3)))
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10))

    logging.info("Model summary %s:", model.summary())
    return model

def train_model(model, X, y):
    logging.info("Training model...")
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    history = model.fit(X, y, epochs=10, 
                        validation_split=0.2)
    return history


def evaluate_model(model, X_test, y_test):
    test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
    logging.info("Model accuracy of test data: %f.2%%", test_acc)
    return {"Loss": test_loss, "accuracy": test_acc}

def run_model_training(vis=False, **kwargs):

    tf_api = kwargs.get("tf_api", "sequential")
    X_train, X_test, y_train, y_test = get_data()
    model = build_model(api=tf_api)
    history = train_model(model, X_train, y_train)
    res = evaluate_model(model, X_test, y_test)

    if vis:
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')
        plt.show()
    
    return model, history, res

# def save_model(model):
#     cnn_save_path = Path(MODEL_PATH, "cnntest/1/")
#     if not cnn_save_path.exists():
#         cnn_save_path.resolve().mkdir()
#     tf.saved_model.save(model, str(cnn_save_path))

def get_model():
    model, _, _ = run_model_training(vis=True)
    return model

if __name__ == "__main__":

    core = ov.Core()
    model, history, metrics = run_model_training(vis=True, tf_api="functional")
    output_path = Path(MODEL_PATH, model.name + ".onnx")
    # input_signature = [tf.TensorSpec((None, 32, 32, 3), tf.float32, name='input')]
    input_signature = [tf.TensorSpec((32, 32, 3), tf.float32, name='input')]
    # Use from_function for tf functions
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, output_path=output_path)
    # ov_model = ov.convert_model(model)
    # ov.save_model(ov_model, 'cnn_test_model.xml')
