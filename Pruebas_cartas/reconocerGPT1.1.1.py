import cv2
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import os

# Cargar el modelo de redes neuronales
model = load_model('card_classifier.h5')

# Cargar las rutas de las imágenes desde el archivo CSV
data = pd.read_csv('cartas_etiquetadas.csv')

# Definir las clases de cartas según el orden en que se entrenó el modelo
classes = ['As', 'Dos', 'Tres', 'Cuatro', 'Cinco', 'Seis', 'Siete', 'Ocho', 'Nueve', 'Diez', 'J', 'Q', 'K']

# Función para preprocesar la imagen antes de la predicción
def preprocess_image(image):
    # Redimensionar la imagen a un tamaño adecuado para el modelo
    resized_image = cv2.resize(image, (224, 224))
    # Normalizar los valores de píxeles entre 0 y 1
    normalized_image = resized_image / 255.0
    # Convertir la imagen a un tensor
    tensor_image = img_to_array(normalized_image)
    # Agregar una dimensión extra para el batch
    tensor_image = np.expand_dims(tensor_image, axis=0)
    # Devolver la imagen preprocesada
    return tensor_image

# Función para decodificar la predicción en la clase de carta
def decode_prediction(prediction):
    # Obtener el índice de la clase con mayor probabilidad
    predicted_index = np.argmax(prediction)
    # Obtener la clase correspondiente al índice
    predicted_class = classes[predicted_index]
    # Devolver la clase predicha
    return predicted_class

# Iterar sobre las imágenes y realizar la detección de cartas
for index, row in data.iterrows():
    # Construir la ruta de la imagen utilizando os.path.join()
    img_path = os.path.join(row[r'C:\Users\Juan\Desktop\Cartas_BD'])
    
    # Cargar la imagen
    img = cv2.imread(img_path)
    
    # Preprocesar la imagen
    img = preprocess_image(img)
    
    # Realizar la predicción utilizando el modelo
    prediction = model.predict(img)
    
    # Decodificar la predicción
    predicted_class = decode_prediction(prediction)
    
    # Mostrar el resultado
    print(f"Carta detectada en la imagen {img_path}: {predicted_class}")
    # Aquí puedes agregar lógica adicional, como guardar la imagen con la detección o mostrarla en una ventana
