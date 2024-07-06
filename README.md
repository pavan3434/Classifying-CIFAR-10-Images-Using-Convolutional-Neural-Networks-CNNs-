# Classifying-CIFAR-10-Images-Using-Convolutional-Neural-Networks-CNNs
This project aims to classify images from the CIFAR-10 dataset using Convolutional Neural Networks (CNNs). The CIFAR-10 dataset contains 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is divided into 50,000 training images and 10,000 test images.

The process begins with loading the CIFAR-10 dataset using Keras' built-in dataset utilities. Initial exploration of the dataset includes displaying the shapes of the training and test sets to understand their structure. A few sample images from the training set are visualized to provide a quick overview of the data.

To prepare the data for training, the pixel values of the images are converted to float type and normalized to a range of 0 to 1. The target class labels are then converted to one-hot encoding using Keras' to_categorical function, facilitating multi-class classification.

The CNN model is constructed using Keras' Sequential API. The architecture includes:

Three convolutional layers with ReLU activation and varying filter sizes.
MaxPooling layers to reduce spatial dimensions.
A Flatten layer to convert the 2D matrices to a 1D vector.
Dense layers with ReLU activation for further processing.
A final Dense layer with softmax activation for outputting probabilities across the 10 classes.
The model is compiled using the Adam optimizer and categorical cross-entropy loss function, chosen for their effectiveness in multi-class classification tasks. The training process involves fitting the model to the training data with validation on the test data. The model is trained for 10 epochs with a batch size of 64.

Post-training, the loss curves for training and validation sets are plotted to visualize the model's learning process. The model's accuracy on the test set is evaluated, and predictions are made on the test images. The predicted labels are compared with the actual labels, and a confusion matrix is generated to assess the model's performance.

Finally, a few test images along with their predicted and actual labels are displayed to provide a visual comparison of the model's predictions against the true labels.

By systematically preprocessing the data, constructing a CNN model, and evaluating its performance, this project demonstrates the application of CNNs in image classification tasks and sets the stage for further enhancements and experimentation.
