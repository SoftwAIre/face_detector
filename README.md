# face_detector

# facial verification, facial detection
# Using neural networks 

Facial recognition and facial verification with deep learning
+ Increased F1 score by 30% using Deep Learning Techniques such as CNNs, DNNs, and Transfer Learning
+ Graphed training and validation accuracy to visualize the accuracy and loss fit to prevent overfitting
+ Tested facial verification using Euclidean and Cosine distances metrics on photo similarity matrices

### Obtain
Using a kaggle DataSet we imported images from around the web for close ups of faces. We collected 3064 number of images. In one folder we have 3064 number of close up human face images from a Kaggle DataSet. In the other folder we have 1377 number of non-human face as well and miscellaneous images. 
![](https://github.com/Chris-Manna/face_detector/blob/master/normal_face.png)

### Scrubbing and Cleaning the DataSet
Once we have the pictures we want to use to train our models we need to prepare them so that they are all the same size and dimensions. For this section, we used the keras.preprocessing.image library to rescale the photos so that they are all the same size, contain the same scaled information representing each pixel, put them in the right directory. 

![](https://github.com/Chris-Manna/face_detector/blob/master/bin_face.png)

We split the images so that we may train our Supervised Machine Learning classification models.

### Densely Connected Network (DCN)
We used Densely connected Networks with tensorflow, adding two hidden layers where we tested out different numbers of nodes and activation functions. We used sgd as an optimizer and binary_crossentropy for thirty epochs which yielded a 99.86% accuracy. 

Then we visualized the training and validation loss functions and compared them against the training and validation accuracy to ensure we would not overfit. We overfit at 60 epochs so we went down to 10 epochs.

![](https://github.com/Chris-Manna/face_detector/blob/master/Densley%20Connected%20Network%20Visualize%20Training:Validation%20Loss.png)

We scored an F1 Score of 99.84%

### Convolutional Neural Network 
While the DCN was free floating nodes in a neural network, we use the convolutional neural network to input sequential layers, mixing in different ways for to aggregate the weighted results of each layer. We visualized the results and got an F1 score of 100%
![](https://github.com/Chris-Manna/face_detector/blob/master/Convolutional%20Neural%20Network:%20Vis%20Train:Val%20Loss.png)

### Drop-out Regularization
From our last Neural Network, sometimes what can happen is that each node will begin to become very fixed in teh way they interpret different portions of a picture - almost like micro overfitting. To ward against this version of overfitting, we introduced Drop-Out Regularization. Drop-Out Regularization randomly removes the value from a certain percentage of the nodes you are using in your neural network. We chose to remove 50% of the nodes from the subsequent layer. Doing this also allows for a more robust interpreting neural network. 

![](https://github.com/Chris-Manna/face_detector/blob/master/DropOut%20Regularization%20vis.png)

Our training accuracy went down and our F1 scores went down, which does not necessarily mean somehting bad happened. In fact, it may mean that we have a more robust, neural network that is no longer overfitting. 

### Transfer Learning
These programs are pretty cool to tinker with, however the number of layers we are using can not compare to Deep Learning layers. While our neural networks have been run on about 4000 facial images, we are exploring Transfer Learning because these Deep Learning Neural Networks have been run over millions of pictures. When the weights have been tuned on millions of pictures, the weights of each of these nodes have had the opportunity to build a solid foundation and their weights are robust to capture nuances of the target wihtin the photos. 



With Transfer Learning you import all the weights from the Deep Learning process done by someone else and discard the last layer of the neural network. You use the last layer to classify and differentiate your target in the images. 

##### MobileNet
![](https://github.com/Chris-Manna/face_detector/blob/master/TransferLearning:MobileNetConfusionMatrix.png)

##### InceptionNet
![](https://github.com/Chris-Manna/face_detector/blob/master/InceptionNetConfusionMatrix.png)

##### VGG16
![](https://github.com/Chris-Manna/face_detector/blob/master/VGG16ConfusionMatrix.png)


In each of these dense neural networks, we used first few dense layers and for the final layer, we used a sigmoid function to determine if the image showed a face or not. VGG16 got the best results as it is geared most towards detecting human faces. 


### Deep face Verification with Keras
This was not a great face detector, let alone facial verification

# Next Steps
- Incorporate video
- Facial Verification





