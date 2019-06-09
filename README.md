# Face Detector
Achieved 98% accuracy predicting if a face is in images through the use of Transfer Learning. 

Detect Faces
+ Use Deep Learning techniques such as CNNs, DNNs, and Transfer Learning.
+ Graph training and validation accuracy to visualize the accuracy loss to prevent overfitting.
+ Test facial verification using Euclidean and Cosine distance metrics on photo similarity matrices.

Libraries/Tools used:
Tensorflow, keras, transfer learning

keras libraries: keras.preprocessing.image
Transfer learning libraries: MobileNet, InceptionNet, VGG16

### Obtain
Import images:
- 3064 pictures of human faces from [faces_data Kaggle DataSet](https://www.kaggle.com/gasgallo/faces-data). 
- 1377 non-human face as well and miscellaneous images. (From where?)
![](https://github.com/Chris-Manna/face_detector/blob/master/normal_face.png)

![]()
Image from not human images

### Scrub and Clean the DataSet
- Rescale images to the same size and dimensions that contain scaled representation of pixels. 
- Split the images so that we may train our Supervised Machine Learning classification models.

Tools used: 
- Rescale the photos with keras.preprocessing.image library

![](https://github.com/Chris-Manna/face_detector/blob/master/bin_face.png)

# Training Models
### Densely Connected Network (DCN)
- Adding hidden layers, testing different numbers of nodes and activation functions. 
- Two hidden layers
- Optimize using SGD, and binary_crossentropy for thirty epochs which yielded a 99.86% accuracy. 
Tools used: Tensorflow, SGD, binary_crossentropy


### Convolutional Neural Network 
- Achieving an F1 score of 100% - likely overfitting here.
- Using convolutional neural network to input sequential layers.
- Mixing node equations in different ways for each layer. 
![](https://github.com/Chris-Manna/face_detector/blob/master/Convolutional%20Neural%20Network:%20Vis%20Train:Val%20Loss.png)

### Drop-out Regularization - Addressing overfitting
- Removing 50% of the nodes from the subsequent layer allowing for a more robust interpretation of neural network. 
- Training accuracy went down and our F1 scores went down.  
- Achieved more robust, neural network.

![](https://github.com/Chris-Manna/face_detector/blob/master/DropOut%20Regularization%20vis.png)

##### Transfer Learning
- While neural networks run in this program so far have been trained on 4000 images, Transfer Learning models used Deep Learning Neural Networks that have been trained on millions of images. 
- When the models have been tuned on millions of pictures, the weights of each of these nodes have captured robust nuances of the intended target wihtin the photos. 

With Transfer Learning you import all the weights from the Deep Learning process done by someone else and replace the last layer of the neural network to detect what you want. 
You use the last layer to classify and differentiate your target in the images. 

##### MobileNet
![](https://github.com/Chris-Manna/face_detector/blob/master/TransferLearning:MobileNetConfusionMatrix.png)

##### InceptionNet
![](https://github.com/Chris-Manna/face_detector/blob/master/InceptionNetConfusionMatrix.png)

##### VGG16 here [](https://arxiv.org/abs/1704.04861)
![](https://github.com/Chris-Manna/face_detector/blob/master/VGG16ConfusionMatrix.png)

In each of these dense neural networks, we used first few dense layers and for the final layer, 
- Use a sigmoid function to determine if the image showed a face or not. 

# Evaluating Models: 
Achieved an F1 Score of 99.84%
- Compared training and validation loss functions against training and validation accuracy to ensure good fit. 
- Overfit at 60 epochs, went down to 10 epochs.

![](https://github.com/Chris-Manna/face_detector/blob/master/Densley%20Connected%20Network%20Visualize%20Training:Validation%20Loss.png)


Summary: 
Transfer Learning library VGG16 got the best results as it is geared most towards detecting human faces. 

# Next Steps
- Incorporate video
- Facial Verification
