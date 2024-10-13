This project is a convolutional neural network (CNN) model for classifying traffic sign images. It uses Keras with TensorFlow as the backend, and applies preprocessing techniques such as grayscale conversion, histogram equalization, and data augmentation. The model is trained on a dataset of traffic signs with corresponding class labels, and can be used for traffic sign recognition.

Features :

Image Preprocessing: Converts images to grayscale, applies histogram equalization, and normalizes the pixel values.

CNN Architecture: Uses multiple convolutional layers, pooling, dropout, and fully connected layers to classify traffic sign images.

Data Augmentation: Uses techniques like shifting, zooming, shearing, and rotation to augment the training data and improve model robustness.

Evaluation: Evaluates the model on a separate test dataset and reports accuracy.

Model Saving: Trained models are saved for future inference.
