# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pathlib

#Establece una semilla para replicar resultados
tf.random.set_seed(17)

main_path = "C:/Users/cesar/Final/"

#Descomprime el archivo de cartas.zip
"""zip_ref = zipfile.ZipFile(main_path + 'cards.zip')
zip_ref.extractall()
zip_ref.close()"""

#Nos da información sobre las carpetas encontradas y la cantidad de archivos en ellas.
for dirpath, dirnames, filenames in os.walk(main_path + 'cards'):
  print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

#Esto genera un arreglo del nombre de las carpetas de los datos del set
data_dir = pathlib.Path(main_path + "cards/train")
class_names = np.array(sorted(item.name for item in data_dir.glob("*")))
print(class_names)

#Data augmentation: Las imágenes del set van a generar nuevas pero ya sean rotadas o estiradas, esto con el
#propósito de mejorar la detección de los objetos. Esta es la definición del objeto que se va a aplicar
data_augmented = ImageDataGenerator(rescale = 1/255., #Normalización
                                          rotation_range = 0.3, #Rotación de imágen
                                          zoom_range = 0.2, #Amplificación de la imagen
                                          width_shift_range = 0.2, #Estiramiento en lo ancho
                                          height_shift_range = 0.2, #Estiramiento en lo largo
                                          horizontal_flip = True) #Rotación horizontal

#Se aplica la configuración del data augmentation a los datos de entrenamiento y de prueba
train_data = data_augmented.flow_from_directory(directory = main_path + "cards/train",
                                                                target_size = (150,132), #Tamaño del set de imágenes
                                                                batch_size = 32, #Tamáño de paquete de imágenes
                                                                class_mode = "categorical", #Esto va con el tipo de red que se va a armar
                                                                shuffle = True) #Desordenar los datos, muy importante
test_data = data_augmented.flow_from_directory(directory = main_path + "cards/test",
                                                                target_size = (150,132),
                                                                batch_size = 32,
                                                                class_mode = "categorical",
                                                                shuffle = True)

#Modelo convolucional
model_cnn = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters = 10, #Cantidad de kernels de la primera capa
                           kernel_size = 3, #Tamaño de los kernels
                           activation = "relu", #Tipo de activación de la neurona
                           input_shape = (150,132,3)), #Tamaño de entrada, debe ser el mismo tamaño que de nuestras imágenes
    tf.keras.layers.Conv2D(10,3, activation = "relu"), #(Forma resumida de la capa anterior, sin declarar explícitamente el nombre de las entradas)
    tf.keras.layers.Conv2D(10,3, activation = "relu"),
    tf.keras.layers.MaxPool2D(), #Capa MaxPool, para 'resumir' la imagen en sus píxeles más importantes
    tf.keras.layers.Conv2D(10,3, activation = "relu"),
    tf.keras.layers.Conv2D(10,3, activation = "relu"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(10,3, activation = "relu"),
    tf.keras.layers.Conv2D(10,3, activation = "relu"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(53, activation = "softmax") #Capa de salida, en esta se debe de especificar la cantidad de clases que se tienen
    ], name = "cnn_model") #Nombre del modelo (opcional)

model_cnn.compile(loss = "categorical_crossentropy", #Tipo de perdida que va a calcular la red, que para este caso de salidas de enteros, es la Categorical_Crossentropy
                  optimizer = tf.keras.optimizers.Adam(), #Optimizador, encargado de los modificar los pesos y sesgos del modelo tras cada época
                  metrics = ["accuracy"]) #Métricas, para saber qué tan bien va nuestro modelo

history_mc = model_cnn.fit(train_data, #Datos de entremiento
                           epochs = 15, #Épocas: cantidad de veces que el modelo va a recorrer todas las imágenes de entrenamiento
                           steps_per_epoch = len(train_data), #Pasos por época: como las imágenes van en paquetes de 32, nos saltamos esa cantidad de imágenes
                           validation_data = test_data, #Set de validación: para saber qué tan bien lo hace con los datos de los que no se está entrenando
                           validation_steps = len(test_data),
                           callbacks = tf.keras.callbacks.EarlyStopping(patience = 3, monitor = "val_loss")) #Callbacks: este en específico, detiene el entrenamiento si no está mejorando lo suficientemente rápido

model_cnn.save(main_path + "cards_model")

