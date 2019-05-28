# face_detector 
Achieved 98% accuracy detecting faces in images through the use of transfer learning. 

Face detection
+ Using Deep Learning techniques such as CNNs, DNNs, and Transfer Learning.
+ Graphing training and validation accuracy to visualize the accuracy loss to prevent overfitting.
+ Test facial verification using Euclidean and Cosine distance metrics on photo similarity matrices.

Libraries/Tools used:
Tensorflow, keras.preprocessing.image, transfer learning
Transfer learning libraries: MobileNet, InceptionNet, VGG16

### Obtain
Import images for close up of faces.
- 3064 pictures of human faces from a Kaggle DataSet. 
- 1377 non-human face as well and miscellaneous images. (From where?)
![](https://github.com/Chris-Manna/face_detector/blob/master/normal_face.png)
![]() 
Image from not human images

### Scrub and Clean the DataSet
- Prepare images so that they are all the same size and dimensions that contain the same scaled representation of pixels. 
- Split the images so that we may train our Supervised Machine Learning classification models.
Tools used: 
- Rescale the photos with keras.preprocessing.image library

![](https://github.com/Chris-Manna/face_detector/blob/master/bin_face.png)

# Training Models
### Densely Connected Network (DCN)
- Adding two hidden layers where we tested out different numbers of nodes and activation functions. 
- Free-floating layers that "evolve" to give the right response. 
- Optimize using SGD, and binary_crossentropy for thirty epochs which yielded a 99.86% accuracy. 
Tools used: Tensorflow, SGD, binary_crossentropy

# Evaluating Models: 
Achieved an F1 Score of 99.84%
- Visualized the training and validation loss functions and compared them against the training and validation accuracy to ensure we would not overfit. 
- Overfit at 60 epochs so we went down to 10 epochs.

![](https://github.com/Chris-Manna/face_detector/blob/master/Densley%20Connected%20Network%20Visualize%20Training:Validation%20Loss.png)

### Convolutional Neural Network 
- Achieving an F1 score of 100% - overfitting here. 
- Using convolutional neural network to input sequential layers.
- Mixing in different ways for to aggregate the weighted results of each layer. 
![](https://github.com/Chris-Manna/face_detector/blob/master/Convolutional%20Neural%20Network:%20Vis%20Train:Val%20Loss.png)

### Drop-out Regularization
- Addressing overfitting, we introduced Drop-Out Regularization. Drop-Out Regularization randomly removes the value from a certain percentage of the nodes you are using in your neural network. 
- Removing 50% of the nodes from the subsequent layer allowing for a more robust interpretation of neural network. 

![](https://github.com/Chris-Manna/face_detector/blob/master/DropOut%20Regularization%20vis.png)

- Training accuracy went down and our F1 scores went down. 
- Achieved more robust, neural network. 

### Transfer Learning
- While neural networks run in this program so far have been run on 4000 facial images the Transfer Learning models used used Deep Learning Neural Networks that have been trained on millions of images. 
- When the models have been tuned on millions of pictures, the weights of each of these nodes have captured robust nuances of the intended target wihtin the photos. 

With Transfer Learning you import all the weights from the Deep Learning process done by someone else and discard the last layer of the neural network. 
You use the last layer to classify and differentiate your target in the images. 

### MobileNet
![](https://github.com/Chris-Manna/face_detector/blob/master/TransferLearning:MobileNetConfusionMatrix.png)

### InceptionNet
![](https://github.com/Chris-Manna/face_detector/blob/master/InceptionNetConfusionMatrix.png)

### VGG16
![](https://github.com/Chris-Manna/face_detector/blob/master/VGG16ConfusionMatrix.png)


In each of these dense neural networks, we used first few dense layers and for the final layer, we used a sigmoid function to determine if the image showed a face or not. VGG16 got the best results as it is geared most towards detecting human faces. 

# Next Steps
- Incorporate video
- Facial Verification

