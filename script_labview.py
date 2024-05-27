#Script de Python para la integración en LabVIEW
#Este .py se le va a mandar a llamar una función, como solo hay una, se manda a llamar a 'main', y se les da los parámetros
#(LabVIEW no sabe qué tipo de datos recibe ni cuántos son, depende del programador enviar los datos correctamente)
def main(card_path, model_path):
    import tensorflow as tf
    from matplotlib.image import imread

    #Lectura de imágen
    img = imread(card_path)
    #Nombre de todas las clases en el orden que se agregaron a la red
    class_names = [0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21,
     22, 23, 24, 25, 26, 27, 28, 29, 3, 30, 31, 32, 33, 34, 35,
     36, 37, 38, 39, 4, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
     5, 50, 51, 52, 6, 7, 8, 9]
    
    #Carga del modelo
    model = tf.keras.models.load_model(model_path)
    
    #Predicción del modelo
    predict = tf.squeeze(model.predict(tf.expand_dims(img, axis=0)))
    
    #Recibiremos el índice del número mayor en todo el arreglo de probabilidades que nos da como predicción la red
    return class_names[tf.math.argmax(predict)]