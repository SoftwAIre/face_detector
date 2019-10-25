# Face Detector

Detect faces in images through the use of Transfer Learning. 
+ Use Deep Learning techniques such as CNNs, DNNs, and Transfer Learning.
+ Graph training and validation accuracy to visualize the accuracy loss to prevent overfitting.
+ Test facial verification using Euclidean and Cosine distance metrics on photo similarity matrices.

Libraries: keras.preprocessing.image
Densely Connected Networks, Convolutional Neural Networks
Transfer Learning Libraries: MobileNet, InceptionNet, VGG16

# Obtain
Import images:
- 3064 pictures of human faces from [faces_data Kaggle DataSet](https://www.kaggle.com/gasgallo/faces-data). 
![](https://github.com/Chris-Manna/face_detector/blob/master/normal_face.png)

- 1377 non-human faces and miscellaneous images.

# Scrub and Clean the DataSet
- Rescale images to the same dimensions with keras.preprocessing.image library. 
- Split the images so that we may train our Supervised Machine Learning classification models.

![](https://github.com/Chris-Manna/face_detector/blob/master/bin_face.png)

# Training Models
### Densely Connected Network (DCN)
- Adding two hidden layers, testing different numbers of nodes and activation functions. 
- Optimize using SGD, and binary_crossentropy for thirty epochs which yielded a 99.86% accuracy. 
Tools used: Tensorflow, SGD, binary_crossentropy

### Convolutional Neural Network 
- Achieve an F1 score of 100% - likely overfitting here.
- Use convolutional neural network to input sequential layers.
- Mix node equations in different ways for each layer. 
![](https://github.com/Chris-Manna/face_detector/blob/master/Convolutional%20Neural%20Network:%20Vis%20Train:Val%20Loss.png)

### Drop-out Regularization - Addressing overfitting
- Remove 50% of the nodes from the subsequent layer allowing for a more robust interpretation of neural network. 
- Train accuracy went down and our F1 scores went down.  
- Achieve more robust, neural network.

![](https://github.com/Chris-Manna/face_detector/blob/master/DropOut%20Regularization%20vis.png)

# Transfer Learning
Using Transfer Learning, replace the last layer of the neural network to classify and differentiate your target in the images. 
- Transfer Learning models used Deep Learning Neural Networks that have been trained on millions of images. 
Explanation: Models have been tuned on millions of pictures, the weights of each of these nodes have captured robust nuances of the intended target within the photos.


### MobileNet
![](https://github.com/Chris-Manna/face_detector/blob/master/TransferLearning:MobileNetConfusionMatrix.png)

### InceptionNet
![](https://github.com/Chris-Manna/face_detector/blob/master/InceptionNetConfusionMatrix.png)

### VGG16 here [](https://arxiv.org/abs/1704.04861)
![](https://github.com/Chris-Manna/face_detector/blob/master/VGG16ConfusionMatrix.png)


- Use a sigmoid function to determine if the image showed a face or not for the last layers. 

# Evaluating Models: 
- Achieve an F1 Score of 99.84%
- Compare training and validation loss functions against training and validation accuracy to ensure good fit. 
- Fit at 60 epochs, went down to 10 epochs.

![](https://github.com/Chris-Manna/face_detector/blob/master/Densley%20Connected%20Network%20Visualize%20Training:Validation%20Loss.png)

# Implement
During live display, get individual images and test individual images using the model.

# Summary
Transfer Learning library VGG16 got the best results.

# Next Steps
- Incorporate video.
- Facial Verification.
- Implement on Raspberry Pi.
