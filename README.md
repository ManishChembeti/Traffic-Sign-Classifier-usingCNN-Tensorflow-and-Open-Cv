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
 
Localization: Detect and localize where in an input image/frame a traffic sign is.

Recognition: Take the localized ROI and actually recognize and classify the traffic sign.

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


****CODE: with detailed explanation of all library and functions used 3.1 Main.py (FOR GUI) OF OUR PROGRAM****


import tkinter as tk
import---> get all system dependencies 
from tkinter import filedialog
#tkinter—GUI RUNNING OF PYTHON
from tkinter import *
from PIL import ImageTk, Image
#PIL- IMAGE MANIPULATION,READING
import numpy
#NUMPY - NumPy is a library for the Python programming language,
adding support for large, multi-dimensional arrays and matrices
import tensorflow as tf
#TENSORFLOW -TensorFlow is an open source library for numerical
computation and large-scale machine learning. TensorFlow bundles 
together a slew of machine learning and deep learning (aka neural 
networking) models and algorithms and makes them useful by way of
a common metaphor.
#load the trained model to classify sign
from keras.models import load_model
model = tf.keras.models.load_model(r'C:\Projects\Traffic Sign
Classifier\traffic_sign_classifier.h5')
Keras is actually included as part of TensorFlow now.
Keras is a neural network library while TensorFlow is the open-source library.
Keras is the most used deep learning framework
 
#dictionary to label all traffic signs class. classes = { 1:'Speed limit (20km/h)', 2:'Speed limit (30km/h)',
3:'Speed limit (50km/h)', 4:'Speed limit (60km/h)', 5:'Speed limit (70km/h)', 6:'Speed limit (80km/h)',
7:'End of speed limit (80km/h)', 8:'Speed limit (100km/h)', 9:'Speed limit (120km/h)', 10:'No passing',
11:'No passing veh over 3.5 tons', 12:'Right-of-way at intersection', 13:'Priority road',
14:'Yield',
15:'Stop',   16:'No vehicles',
17:'Veh > 3.5 tons prohibited', 18:'No entry',
19:'General caution', 20:'Dangerous curve left', 21:'Dangerous curve right', 22:'Double  curve', 23:'Bumpy road',
24:'Slippery road'
26:'Road work', 27:'Traffic signals', 28:'Pedestrians', 29:'Children crossing', 30:'Bicycles crossing', 31:'Beware of ice/snow', 32:'Wild animals crossing',
33:'End speed + passing limits', 34:'Turn right ahead',
35:'Turn left ahead', 36:'Ahead only',
37:'Go straight or right', 38:'Go straight or left', 39:'Keep right',
40:'Keep left', 41:'Roundabout mandatory', 42:'End  of  no  passing',
43:'End no passing veh > 3.5 tons' } #initialise GUI
top=tk.Tk() top.geometry('800x600')
top.title('Traffic sign classification') top.configure(background='#72b0ff') label=Label(top,background='#72b0ff',  font=('arial',15,'bold')) sign_image = Label(top)
 
def classify(file_path):
global label_packed
image = Image.open(file_path) image = image.resize((30,30))
image = numpy.expand_dims(image, axis=0) image = numpy.array(image)
pred = model.predict_classes([image])[0] sign = classes[pred+1]
print(sign)  label.configure(foreground='#011638', text=sign) def show_classify_button(file_path):
--------->  WHEN  U  PRESS  CLASSIFY  IN  GUI  OUTPUT  (WINDOW  INTERFACE  CODE)
classify_b=Button(top,text="Classify  Image",command=lambda: classify(file_path),padx=10,pady=5)--->  FILE  PATH classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))---->  CROP  FILE  30X30  EXPAND  INTO NUMPY  ARRAY-  PUSH  INTO  H5  FILE
classify_b.place(relx=0.79,rely=0.46) def upload_image():
try: file_path=filedialog.askopenfilename() uploaded=Image.open(file_path)
uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25))) im=ImageTk.PhotoImage(uploaded)
sign_image.configure(image=im) sign_image.image=im  label.configure(text='') show_classify_button(file_path)->CLASSIFY BUTTON
except:
pass
upload=Button(top,text="Upload an image",command=upload_image,padx=10,pady=5) upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))  ----------->  IMAGE  UPLOADING upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True) label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Know Your Traffic Sign",pady=20, font=('arial',20,'bold')) heading.configure(background='#72b0ff',foreground='#364156') heading.pack()
top.mainloop()
---------->  PILLOW  LIBRARY:  Python  Imaging  Library  is  a  free  and  open-source additional  library  for  the  Python  programming  language  that  adds  support  for opening,  manipulating,  and  saving  many  different  image  file  formats





