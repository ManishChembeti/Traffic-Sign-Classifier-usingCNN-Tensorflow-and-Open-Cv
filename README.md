# Traffic-Sign-Classifier-usingCNN-Tensorflow-and-Open-Cv

Today’s more advanced technologies are furthering our goals and helping with automation in every field making the need for a human in those areas invalid, because a human is prone to making mistakes, but a machine in his/her place would certainly be more efficient, both in terms of speed and accuracy. Technologies such as Deep Learning and Machine Learning have evolved greatly in this time.

This technology helps to teach machines to learn on their own instead of having to program every single action and possibility. So, this project helps us to use techniques like this such as convolution neural networks, keras, Tensorflow, etc. and implement them so as to help the self-driving cars to be able to perceive traffic signs and react according to the input received.

In this project, we will build a deep neural network model that can classify traffic signals present in the image into different categories. With this model, we are able to understand and read traffic signals, which is very important to self-driving cars because it can otherwise lead to road accidents.

The main objective of this project is to develop a product which would help people learn about one of most underrated, yet very import part of our daily life, a traffic sign. This model has been made using deep learning libraries Tensorflow and its high level API, Keras. The objective of this model is to attain an accuracy so strong that an individual should be able to use our product without any hesitation.

In the past and recent times, there have been many road accidents where the main reason for these being inadequate knowledge of road and traffic signs. Even though speed is one of the key issue for the cause of such atrocities, in a survey, it was found out that the second most heard reason was an individual not knowing what a particular traffic sign meant.
Our team strongly believes that the product that we have developed would help individuals learn these signs intuitively, especially the adolescence of 21st century, who also stay and live around technology, which is growing faster than ever.

Our project focuses on detecting traffic signs, when provided an image to it through deep learning, image processing through OpenCV and a convenient UI is has been developed in Python GUI using Tkinter.

Initially after downloading the data set, the whole training of the model has been done in Anaconda. The images were initially divided into training and testing sets and were loading into the notebook using the OS module in python.
Then the required dependencies have been installed for training the network. Then the images for training have been loaded and are reshaped to have the same shape, to help the CNN model for training.

Then the outputs have been converted to categorical values. We have developed our own architecture for this project. Our architecture consisted of 2 sets of 2 convolution layers followed a max pooling layer and a dropout ratio has also been provided.

A validation split was also mentioned at the starting of training of the model, which helped us understand how our model was working after every epoch. After the whole model was trained, we also performed a test using the test set, and our trained model achieved an accuracy of 96%, which is considered very strong, taking the accuracy into account.

Then we have saved our model as h5 file in order to use it in our GUI. We have developed a python GUI using Tkinter, where a user can upload an image of a traffic sign and we predict that sign and display it back to the user.

***DESIGN***

For the design part we have made an architecture after doing research on various other architectures like Alex net, VGG16 and VGG19.The type of network that we have used in our is the very well-known CNN.
The research on these architectures and network structures gave us a proper insight into how to make our own architecture.
This research gave us an idea of how to put convolution layer and maximum pooling layer as well as the drop out values in order to reduce the computational power need as well as increase of accuracy. The basic functionality of CNN is given in figure.


![image](https://user-images.githubusercontent.com/59841174/119186637-ff3ccc00-ba95-11eb-909d-72a627760334.png)


A convolutional neural network consists of an input and an output layer, as well as multiple hidden layers. The hidden layers of a CNN typically consist of a series of convolutional layers that convolve with a multiplication or other dot product. The activation function is commonly a RELU layer, and is subsequently followed by additional convolutions such as pooling layers, fully connected layers and normalization layers, referred to as hidden layers because their inputs and outputs are masked by the activation function and final convolution.

When programming a CNN, the input is a tensor with shape (number of images) x (image height) x (image width) x (image depth). Then after passing through a convolutional layer, the image becomes abstracted to a feature map, with shape (number of images) x (feature map height) x (feature map width) x (feature map channels).

A convolutional layer within a neural network should have the following attributes: Convolutional kernels defined by a width and height (hyper-parameters).
The number of input channels and output channels (hyper-parameter).
The depth of the Convolution filter (the input channels) must be equal to the number channels (depth) of the input feature map.


******Our traffic sign dataset:******

The data set we have decided to use for our project was the GTSRB- German Traffic Sign Detection Benchmark .This is one of the most renowned datasets for traffic signs in websites like kaggle. This data set has more that 40 classes of images and 50000 images for training, validation and testing purposes. We have divided the data set into training, validation and testing set, which further helped us in understanding how well our architecture was working.
In the real-world, traffic sign recognition is a two-stage process:
 
****Localization:**** Detect and localize where in an input image/frame a traffic sign is.

****Recognition:**** Take the localized ROI and actually recognize and classify the traffic sign.

Deep learning object detectors can perform localization and recognition in a single forward-pass of the network.


****DATASET DOWNLOAD WHERE AND HOW?****

From here we’ll download the GTSRB dataset from online. Simply click the “Download (300MB)” button in the website menubar and follow the prompts to sign in using one of the third party authentication partners or with your email address. You may then click the “Download (300MB)” button once more and your download will commence as shown.



****STRUCTURAL REPRESENTATION:****

Basically when an image is passed to a model, it is passed through 2 convolution layer followed by a maximum pooling layer of pooling size (2,2)
Maximum pooling layer is used to reduce the dimensions but still retain the details of an image. This set is repeated for two times and then it is flattened and passed to a fully connected dense layer network.

The activation functions used here are rectified linear unit functions, followed by another fully connected layer which runs on a softmax layer in order to predict the class of a traffic sign to which it belongs. The neural network architectural design that has been used in our project is presented in figure 2.


![image](https://user-images.githubusercontent.com/59841174/119189482-bb4bc600-ba99-11eb-9600-5c68aa9014f5.png)
Developed CNN structure


![image](https://user-images.githubusercontent.com/59841174/119189572-d7e7fe00-ba99-11eb-9cef-cb00ff9c09b3.png)
Architecture Table


****4.	DISPLAYING  OUTPUT/RESULTS:****

Open Anaconda.
Activate  tensor  flow  -->  conda  activate  tf

Enter	the	drive	in	which	ur	program	files	and	datasets	are present.

Run  the python file	for the gui along with the execution of datasets  and  model  training  simuntaneously  using  Anaconda.

You get a blue GUI window,asking you to upload an traffic image to  classify.

You press classify and it shows what the respective sign is.


![image](https://user-images.githubusercontent.com/59841174/119191320-18e11200-ba9c-11eb-943d-6422526ef3c6.png)


![image](https://user-images.githubusercontent.com/59841174/119191406-331af000-ba9c-11eb-933a-89ea4a7c4890.png)

![image](https://user-images.githubusercontent.com/59841174/119565893-24984580-bdc8-11eb-8dfe-9c141c35ff0c.png)

****Output Summary****

In our project, we came out with a very nice architecture which attained an exceptional accuracy of 98.8% on the validation set and an accuracy of 96% on the test set. Even the data set used was successfully able to get divided into testing and training sets. This model was saved as a h5 file whose location is further passed to our file containing our GUI, for using our trained model extensively.
We were also successful in developing this GUI, using which, a user can upload an image in our GUI and the user would get a message of what traffic sign it was. The accuracy graph of the model has presented in figure 3.





