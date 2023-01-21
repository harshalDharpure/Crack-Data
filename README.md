# Crack-Data

## Model Description

We use ResNet50V2 model from TensorFlow's Keras library for transfer learning. This model is a version of the ResNet50 model that has been updated with improved architecture and performance.

The code first creates an instance of the ResNet50V2 model with the input shape of (224, 224, 3), which means that the model expects images with the size of 224x224 pixels and 3 color channels (RGB). The include_top=False argument specifies that the final fully-connected layers of the model should not be included, allowing us to add your own layers on top of the pre-trained layers.
Then, it creates a new sequential model and stack a GlobalAveragePooling2D layer and a Dense layer with one unit and sigmoid activation function on top of the pre-trained ResNet50V2 model. This new architecture will use the features learned by the pre-trained model, and fine-tune it on the target dataset.

Finally, the model.summary() function call will display the summary of the model's architecture, including the output shape of each layer and the number of parameters in each layer.

94668760/94668760 [==============================] - 7s 0us/step
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 resnet50v2 (Functional)     (None, 7, 7, 2048)        23564800  
                                                                 
 global_average_pooling2d (G  (None, 2048)             0         
 lobalAveragePooling2D)                                          
                                                                 
 dense (Dense)               (None, 1)                 2049      
                                                                 
=================================================================
Total params: 23,566,849
Trainable params: 23,521,409
Non-trainable params: 45,440
_________________________________________________________________

![download (9)](https://user-images.githubusercontent.com/63531290/212933957-890a6644-acb9-410a-87ec-bbbe512a0b7d.png)

## Accuracy Of the Model

7/7 [==============================] - 2s 121ms/step - loss: 0.1909 - accuracy: 0.9850 - precision: 0.9802 - recall: 0.9900
 
 
##Precision Value - 0.9801980257034302
##Recall Value -  0.9900000095367432
##F1- Score - 0.9655172494444295

 
 ![Screenshot (1271)](https://user-images.githubusercontent.com/63531290/212934954-9b2a073f-8ebd-4039-81ed-f3597ed63b4a.png)

 
 Accuracy: 0.985 is a good value, it means that the model is able to correctly identify cracks in images with a 98.5% rate.

Precision: 0.980 is also a good value, it means that when the model predicts that an image contains a crack, it is correct 98% of the time.

Recall: 0.990 is a good value, it means that the model is able to identify 99% of the images that contain cracks.

while these values are indicative of good performance, they should be interpreted in the context of the specific task and dataset, and should be compared against a suitable baseline. Additionally, it's always a good idea to test the model on a different set of images to ensure its generalization capabilities.

![Screenshot (1274)](https://user-images.githubusercontent.com/63531290/213128906-037f1278-37d3-4360-a988-6ec0fa093bda.png)


An F1 score is a measure of a model's accuracy that takes into account both precision and recall. The F1 score ranges from 0 to 1, with a score of 1 indicating perfect accuracy and a score of 0 indicating complete inaccuracy. A score of 0.9655172494444295 would indicate a relatively high level of accuracy.

## -------------------------------------------------------------------------------------------------------------------------------------------------

# ResNet50 TransferLearning Model
ResNet-50 is a 50-layer convolutional neural network

![image](https://user-images.githubusercontent.com/87661736/213517377-7905dcdb-8e08-44de-8795-92e47d6d2a94.png)

## Accuracy Of the Model
7/7 [==============================] - 2s 85ms/step - loss: 0.0217 - accuracy: 0.9950 - precision: 0.9901 - recall: 1.0000

##Precision Value - 0.9900990128517151
##Recall Value -   1.00000
##F1- Score - 0.9950248771119697

 
 
![image](https://user-images.githubusercontent.com/87661736/213517541-1c409aca-86c1-4bc0-8ada-3473414857eb.png)

 An F1 score is a measure of a model's accuracy that takes into account both precision and recall. The F1 score ranges from 0 to 1, with a score of 1 indicating perfect accuracy and a score of 0 indicating complete inaccuracy. A score of 0.9950248771119697 would indicate a relatively high level of accuracy.
 
![image](https://user-images.githubusercontent.com/87661736/213517848-5ac637b2-7747-4746-a35e-3feed721ece6.png)

## ---------------------------------------------------------------------------------------------------------------------------------------------------


# VGG16 TransferLearning Model

VGG16 is a 16 layer transfer learning architecture
VGG Net uses a deep neural network, so VGG can extract more information from the network as it deals multiple parameters.


![image](https://user-images.githubusercontent.com/87661736/213514493-0dc35d3a-6cd8-4fa2-907e-051664d10409.png)

## Accuracy Of the Model

7/7 [==============================] - 1s 139ms/step - loss: 0.4263 - accuracy: 0.9650 - precision: 0.9697 - recall: 0.9600


##Precision Value - 0.9696969985961914
##Recall Value -   0.9599999785423279
##F1- Score - 0.9648241240708334

 
 ![image](https://user-images.githubusercontent.com/87661736/213514793-6be93354-a4b6-458b-bcfc-fa1206f99cff.png)

while these values are indicative of good performance, they should be interpreted in the context of the specific task and dataset, and should be compared against a suitable baseline. Additionally, it's always a good idea to test the model on a different set of images to ensure its generalization capabilities.
 
 ![image](https://user-images.githubusercontent.com/87661736/213515187-6cb390c6-9e8a-45d2-8e4c-150f683f2c92.png)

 An F1 score is a measure of a model's accuracy that takes into account both precision and recall. The F1 score ranges from 0 to 1, with a score of 1 indicating perfect accuracy and a score of 0 indicating complete inaccuracy. A score of 0.9648241240708334 would indicate a relatively high level of accuracy.
