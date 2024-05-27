Versión de Python utiizada: 3.10.13
Versión de LabVIEW utilizada: 2024 Q1 (64-Bit)

Nota 1: Para que funcione correctamente, esta carpeta debe de estar guardada en el almacenamiento local, no en la nube, debido a que
se han experimentado problemas a guardarlos en carpetas de OneDrive de Windows.

Nota 2: Este modelo fue entrenado para un barja inglesa en particular, no se asegura su funcionamiento con cualquier tipo de baraja.
Información de la baraja: Baraja Inglesa Bicyle estándar (https://a.co/d/3m34g8j)

1. Ejecutar el model_cards.py y cambiar la ruta principal (main_path) a donde se haya guardado esta carpeta. 
Este descomprimirá el zip de cartas y creará el modelo con los datos de este

2. Ejecutar DLOL Cards.vi, hay que cambiar las rutas de los archivos que se especifican abajo del panel frontal (son 4),
Seleccionar la cámara que se va a utilizar

3. Una vez ejecutado, se selecciona con las herramientas del control de la imagen un ROI que encierre a la carta, y posteriormente
se presione el "Update ROI" para solo cargar esa zona en la red neuronal.