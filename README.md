# Model Building, Evaluation and Validation
## Project:  CIFAR-10

## 	Introduction to the project 
CIFAR-10 is an established computer-vision dataset used for object recognition. It is a subset of the 80 million tiny images dataset and consists of 60,000 32x32 color images containing one of 10 object classes, with 6000 images per class. It was collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton There are 50000 training images and 10000 test images.

Computer algorithms for recognizing objects in photos often learn by example. CIFAR-10 is a set of images that can be used to teach a computer how to recognize objects. Since the images in CIFAR-10 are low-resolution (32x32), this dataset can allow researchers to quickly try different algorithms to see what works. Various kinds of convolutional neural networks tend to be the best at recognizing the images in CIFAR-10.

CIFAR-10 is a labeled subset of the 80 million tiny images dataset. When the dataset was created, students were paid to label all of the images.
Our goal is to create an accurate deep learning model that can classify the images using the given data.

We gonna use CNN method. Convolutional neural network(CNN) is a strong algorithm in machine learning area. There are lots of networks around us, but the special property of this is that, it can detect special features of an image or data for each class by itself. It applies certain kernels over each image, scans the matched patterns and extract the contextual features. In every CNN model, image processing functions like edge detection, blurring, sharpening occur in the convolution layers and identifying or memorizing tasks are occured in the fully connected layer portion of model.

## 	How others should get started with your project
### 1. Install

This project requires **Python** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

You will also need to have software installed to run and execute a [Jupyter Notebook](http://jupyter.org/install.html).

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](https://www.anaconda.com/download/) distribution of Python, which already has the above packages and more included. 
## 	How others can run and test the project
In a terminal or command window, navigate to the top-level project directory `boston_housing/` (that contains this README) and run one of the following commands:
```bash
ipython notebook boston_housing.ipynb
```  
or
```bash
jupyter notebook boston_housing.ipynb
```
or open with Juoyter Lab
```bash
jupyter lab
```

This will open the Jupyter Notebook software and project file in your browser.
## 	What libraries and functions are used and why
In this code i using libraries like 
#### TensorFlow:
-TensorFlow is an open-source machine learning library developed by Google.
-It provides high-level APIs for building and training neural networks.
-The TensorFlow library is  used for implementing deep learning models for image classification tasks like CIFAR-10.
#### Scikit-learn:

-Scikit-learn is a general-purpose machine learning library in Python.
-It provides tools for data preprocessing, feature selection, and model evaluation.
-Scikit-learn  used in conjunction with other libraries for building machine learning models on top of traditional algorithms.
#### Matplotlib and Seaborn:

-Used these  visualization libraries in Python that are  used to plot and visualize the performance metrics of models, such as accuracy and loss over training epochs.
#### NumPy:

-NumPy  used for numerical operations and array manipulations, which are common tasks in the preprocessing and analysis of image data.

## 	The results of your model: accuracy and prediction for the given project
![image](https://github.com/monicaavagyan/CIFAR-10/assets/130900080/e4fb56fc-243a-49ad-ac99-7a0ff65237bf)

The accuracy increases over time and the loss decreases over time. However, the accuracy of our validation set seems to slightly decrease towards the end even thought our training accuracy increased. Running the model for more epochs might cause our model to be susceptible to overfitting.

## Predict evaluate
313/313 [==============================] - 10s 30ms/step - loss: 1.0657 - acc: 0.6239
- loss:  0.8886: This is the value of the loss function on the dataset. The loss function is a measure of how well the model is performing. In this context, it's  a classification loss, and a lower value is generally better. 

- acc: 0.6913: This is the accuracy of the model on the dataset. The accuracy is the ratio of correctly predicted instances to the total instances. In this case, 0.6913 indicates an accuracy of approximately 70%.

### Have questions, contact me.
[![LinkedIn](https://img.shields.io/static/v1.svg?label=connect&message=@monica-avagyan&color=success&logo=linkedin&style=flat&logoColor=white&colorA=blue)](https://www.linkedin.com/in/monica-avagyan/)


[![GitHub](https://img.shields.io/static/v1.svg?label=connect&message=@monicaavagyan&color=success&logo=github&style=flat&logoColor=white&colorA=blue)](https://github.com/monicaavagyan)

:email: Feel free to contact me @ [avagyanmonika3@gmail.com](https://mail.google.com/mail/)
