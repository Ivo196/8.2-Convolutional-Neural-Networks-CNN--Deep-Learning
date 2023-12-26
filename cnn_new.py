# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 20:59:13 2023

@author: ivoto
"""

# Redes Neuronales Convolucionales

# Instalar Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Instalar Tensorflow y Keras
# conda install -c conda-forge keras

# Parte 1 - Construir el modelo de CNN 
# Importar Keras y librerías adicionales
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Inicializar la CNN 
classifier = Sequential()

# Paso 1 - Convolucion 
classifier.add(Conv2D(filters = 32, kernel_size = (3, 3), input_shape = (64, 64, 3), activation = 'relu')) 

# Paso 2 - Max Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Una segunda capa de convolucion y max pooling
classifier.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu')) 
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Paso 3 - Flatting
classifier.add(Flatten())

# Paso 4 - Full Connection 
classifier.add(Dense(units = 128, activation = 'relu')) 
classifier.add(Dense(units = 1, activation = 'sigmoid')) 

# Compilar la CNN 
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ["accuracy"])

# Parte 2 - Ajustar la CNN a las imagenes para entrenar
from keras.preprocessing.image import ImageDataGenerator

# Escalado conjunto de train
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# Escalado conjunto de test
test_datagen = ImageDataGenerator(rescale=1./255)

# Carga del conjunto de train
training_dataset = train_datagen.flow_from_directory('dataset/training_set', 
                                                    target_size=(64, 64), #Mantener la proporcion con el algoritomo 
                                                    batch_size=32, 
                                                    class_mode='binary') 

# Carga del conjunto de test
testing_dataset = test_datagen.flow_from_directory('dataset/test_set', 
                                                        target_size=(64, 64),
                                                        batch_size=32,
                                                        class_mode='binary')
#Entrenamos el modelo
classifier.fit(training_dataset, 
               steps_per_epoch=250, #Cantidad de imagenes 
               epochs=25, 
               validation_data=testing_dataset,
               validation_steps=2000)






