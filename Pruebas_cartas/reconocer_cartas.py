import cv2
import numpy as np
import pandas as pd
from keras.models import load_model

# Función para preprocesar las imágenes
def preprocess_image(image_path, target_size=(100, 100)):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, target_size)
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    normalized_image = gray.astype('float32') / 255.0
    return normalized_image

# Cargar el modelo entrenado
model = load_model('card_classifier.h5')
print("Modelo cargado con éxito.")

# Cargar el archivo CSV que contiene las rutas de las imágenes y sus etiquetas
data = pd.read_csv('cartas_etiquetadas.csv')

# Función para reconocer la carta en una imagen
def recognize_card(image_path, model):
    # Preprocesar la imagen
    preprocessed_image = preprocess_image(image_path)
    
    # Agregar una dimensión adicional para que tenga la forma correcta para la entrada del modelo
    preprocessed_image = np.expand_dims(preprocessed_image, axis=-1)
    
    # Realizar la predicción utilizando el modelo
    prediction = model.predict(np.array([preprocessed_image]))
    
    # Decodificar la salida del modelo
    predicted_label = np.argmax(prediction)
    
    return predicted_label

# Crear un diccionario de etiqueta a nombre de la carta para mapear los números a sus nombres
card_names = {
    0: '2', 1: '3', 2: '4', 3: '5', 4: '6', 5: '7', 6: '8', 7: '9', 8: '10',
    9: 'J', 10: 'Q', 11: 'K', 12: 'A', 13: 'Hearts', 14: 'Diamonds', 15: 'Clubs', 16: 'Spades'
}

# Ruta de la imagen que deseas reconocer
image_path = '10_rombos.jpg'

# Reconocer la carta en la imagen y obtener el nombre
predicted_label = recognize_card(image_path, model)
if predicted_label in card_names:
    card_name = card_names[predicted_label]
    print(f"La carta en la imagen es: {card_name}")
else:
    print("No se pudo reconocer la carta en la imagen.")
