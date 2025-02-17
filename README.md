# MNIST Digit Recognition using Artificial Neural Network (ANN)

This project demonstrates the implementation of an Artificial Neural Network (ANN) using TensorFlow and Keras to classify handwritten digits from the MNIST dataset. The model achieves an accuracy of over 99% on the training set and over 98% on the test set.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Confusion Matrix](#confusion-matrix)
- [Saving the Model](#saving-the-model)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [License](#license)

## Introduction
The MNIST dataset is a widely used dataset for image classification tasks. It consists of 60,000 training images and 10,000 test images of handwritten digits (0-9). This project uses a simple ANN to classify these digits.

## Dataset
The dataset is loaded using TensorFlow's `keras.datasets.mnist.load_data()` function. The images are 28x28 grayscale images, and the labels are integers ranging from 0 to 9.

```python
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
```

## Model Architecture
The ANN model consists of the following layers:
1. **Flatten Layer**: Converts the 28x28 image into a 1D array of 784 elements.
2. **Dense Layer**: 300 neurons with ReLU activation.
3. **Dense Layer**: 100 neurons with ReLU activation.
4. **Output Layer**: 10 neurons with Softmax activation (one for each class).

```python
ann = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.Dense(300, activation='relu'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
```

## Training
The model is compiled using the Adam optimizer and trained for 50 epochs. The loss function used is `sparse_categorical_crossentropy`, and the metric used is accuracy.

```python
ann.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
ann.fit(X_train, y_train, epochs=50)
```

## Evaluation
The model is evaluated on the test set, achieving an accuracy of over 98%.

```python
ann.evaluate(X_test, y_test)
```

## Results
- **Training Accuracy**: >99%
- **Test Accuracy**: >98%

## Confusion Matrix
A confusion matrix is generated to visualize the performance of the model on the test set.

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

predictions = ann.predict(X_test).argmax(axis=1)
plt.figure(figsize=(10,10))
sns.heatmap(confusion_matrix(y_test, predictions), annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
```

## Saving the Model
The trained model is saved as an HDF5 file (`image_recognition.h5`) for future use.

```python
ann.save('image_recognition.h5')
```

## Usage
To use the trained model, you can load it using TensorFlow and make predictions on new data.

```python
from tensorflow import keras
model = keras.models.load_model('image_recognition.h5')
predictions = model.predict(X_new)
```

## Dependencies
- Python 3.x
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

Install the required packages using:

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

