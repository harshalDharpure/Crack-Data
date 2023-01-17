# Crack-Data

##Model Description

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

##Accuracy Of the Model

7/7 [==============================] - 2s 121ms/step - loss: 0.1909 - accuracy: 0.9850 - precision: 0.9802 - recall: 0.9900
 Testing Acc :  0.9850000143051147
 Testing Precision  0.9801980257034302
 Testing Recall  0.9900000095367432
 
 
 Accuracy: 0.985 is a good value, it means that the model is able to correctly identify cracks in images with a 98.5% rate.

Precision: 0.980 is also a good value, it means that when the model predicts that an image contains a crack, it is correct 98% of the time.

Recall: 0.990 is a good value, it means that the model is able to identify 99% of the images that contain cracks.

while these values are indicative of good performance, they should be interpreted in the context of the specific task and dataset, and should be compared against a suitable baseline. Additionally, it's always a good idea to test the model on a different set of images to ensure its generalization capabilities.
