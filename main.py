import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

folder = "Digits"   
# # importing data
# data = tf.keras.datasets.mnist
# # Train_test Splitting
# (train_x,train_y),(test_x,test_y)=data.load_data()
# # Normalizing
# x_train= tf.keras.utils.normalize(train_x)
# x_test= tf.keras.utils.normalize(test_x)

# # 1 input layer 2 hidden layer and 1 output layer
# model=tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(28,28))) # Flatten layer is used to convert the 2D image into a 1D vector
# model.add(tf.keras.layers.Dense(128,activation='relu')) # hidden layer 1 # relu is used as an activation function
# model.add(tf.keras.layers.Dense(64,activation='relu')) # hidden layer 2
# model.add(tf.keras.layers.Dense(10,activation='softmax')) # output layer # softmax is used for multi-class classification

# model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#     # adam is an optimization algorithm # sparse_categorical_crossentropy is used for multi-class classification  
# model.fit(x_train,train_y,epochs=10) # epochs is the number of times the model will see the entire dataset

# accuracy=model.evaluate(x_test,test_y)
# loss=model.evaluate(x_test,test_y)
# print("Accuracy:",accuracy)
# print("Loss:",loss)

# model.save('handwritten.keras')



model=tf.keras.models.load_model('handwritten.keras')

for i in range(0,11):
    path = os.path.join(folder, f"{i}.png")
    img=cv.imread(path, cv.IMREAD_GRAYSCALE) # reading the image and converting it to grayscale
    if img is None:
       raise FileNotFoundError(f"Image not found: {i}.png")


    img=cv.resize(img,(28,28)) # resizing the image to 28x28
    img = tf.keras.utils.normalize(img, axis=1)
    img = np.invert(img.astype(np.uint8)) # inverting the image
    img=img.reshape(1,28,28) # reshaping the image to (1,28,28)
    prediction=model.predict(img) # predicting the image
    print("Prediction:",np.argmax(prediction)) # printing the predicted class
    plt.imshow(img[0],cmap=plt.cm.binary) # displaying the image
    plt.show()
