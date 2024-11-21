import tensorflow as tf
from tensorflow import keras

from cv2 import *

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pre_processing as pre_process
import DCT as dct


class Neural_Network:
    def __init__(self):
        fashion_mnist = keras.datasets.fashion_mnist
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = fashion_mnist.load_data()
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']    
        self.train_images = self.train_images / 255.0
        self.test_images = self.test_images / 255.0
        self.prediction = []
        self.imagens_filtradas_cos = []
        self.imagens_filtradas_pb = []
    
    def dysplay_images(self,n_images):
        
        plt.figure(figsize=(10,10))
        for i in range(n_images):
            plt.subplot(5,5,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(self.train_images[i+1000], cmap=plt.cm.binary)
            plt.xlabel(self.class_names[self.train_labels[i+1000]])
        plt.show()
    
    def display_results(self,rows,columns):
        num_images = rows*columns
        plt.figure(figsize=(2*2*columns, 2*rows))
        for i in range(num_images):
            plt.subplot(rows, 2*columns, 2*i+1)
            pre_process.plot_image(i, self.predictions, self.test_labels, self.imagens_filtradas_cos)
            plt.subplot(rows, 2*columns, 2*i+2)
            pre_process.plot_value_array(i, self.predictions, self.test_labels)
        plt.show()

    def build(self):
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(10, activation=tf.nn.softmax)
            ])
        
        model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
            
        model.fit(self.train_images, self.train_labels, epochs=10)
    # test_loss, test_acc = model.evaluate(test_images, test_labels)
        test_loss, test_acc = model.evaluate(self.imagens_filtradas_cos, self.test_labels)
        print('Test accuracy:', test_acc)

        self.predictions = model.predict(self.imagens_filtradas_cos)



if __name__ == "__main__": 
    nn = Neural_Network()
    nn.dysplay_images(25)
    # nn.build()
    # plt.imshow(nn.train_images[4000], cmap="gray")
    # plt.title("Original")
    # plt.show()

    # nn.display_results(4,4)
    # print(type(nn.train_images[27]))
    # pre_process.test_filter(nn.train_images[4000])
    # nn.imagens_filtradas_cos = pre_process.filter_cosseno(nn.test_images)
    # nn.imagens_filtradas_pb = pre_process.filter_cosseno(nn.test_images)
    # nn.build()
    # print(type(nn.imagens_filtradas_cos))
    # print(nn.imagens_filtradas_cos.shape)
    # print(type(nn.test_images))
    # print(nn.test_images.shape)
    # nn.display_results(4,4) 

 