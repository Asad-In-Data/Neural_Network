import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# importing data
data = tf.keras.datasets.mnist
# Train_test Splitting
(train_x,train_y),(test_x,test_y)=data.load_data()
# Normalizing
x_train= tf.keras.utils.normalize(train_x)
x_test= tf.keras.utils.normalize(test_x)

# 1 input layer 2 hidden layer and 1 output layer
model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28))) # Flatten layer is used to convert the 2D image into a 1D vector
model.add(tf.keras.layers.Dense(128,activation='relu')) # hidden layer 1 # relu is used as an activation function
model.add(tf.keras.layers.Dense(64,activation='relu')) # hidden layer 2
model.add(tf.keras.layers.Dense(10,activation='softmax')) # output layer # softmax is used for multi-class classification

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    # adam is an optimization algorithm # sparse_categorical_crossentropy is used for multi-class classification  
model.fit(x_train,train_y,epochs=10) # epochs is the number of times the model will see the entire dataset

accuracy=model.evaluate(x_test,test_y)
loss=model.evaluate(x_test,test_y)
print("Accuracy:",accuracy)
print("Loss:",loss)

model.save('handwritten.model')
