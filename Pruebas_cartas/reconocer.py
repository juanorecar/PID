import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Función para preprocesar las imágenes
def preprocess_image(image_path, target_size=(100, 100)):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, target_size)
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    normalized_image = gray.astype('float32') / 255.0
    return normalized_image

# Directorio que contiene las imágenes de las cartas etiquetadas
directory = r'C:/Users/Juan/Desktop/Cartas_BD/'

# Lista para almacenar las rutas de las imágenes y sus etiquetas
data = []

# Recorrer los directorios y archivos en el directorio de imágenes
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):  # Asegúrate de que solo se incluyan archivos de imagen
            image_path = os.path.join(root, file)
            label = os.path.basename(root)  # La etiqueta es el nombre del subdirectorio actual
            data.append((image_path, label))

# Crear un DataFrame de Pandas con las rutas de las imágenes y sus etiquetas
df = pd.DataFrame(data, columns=['ruta_imagen', 'etiqueta'])

# Dividir los datos en conjuntos de entrenamiento y prueba
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Obtener las rutas de las imágenes y las etiquetas correspondientes para los conjuntos de entrenamiento y prueba
train_image_paths = train_data['ruta_imagen'].values
train_labels = train_data['etiqueta'].values
test_image_paths = test_data['ruta_imagen'].values
test_labels = test_data['etiqueta'].values

# Codificar las etiquetas
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(train_labels)
y_test_encoded = label_encoder.transform(test_labels)

# Preprocesar las imágenes y generar los conjuntos de datos X_train e X_test
X_train = np.array([preprocess_image(image_path) for image_path in train_image_paths])
X_test = np.array([preprocess_image(image_path) for image_path in test_image_paths])

# Definir el modelo de clasificación de cartas utilizando Keras
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))  # Número de clases: len(label_encoder.classes_)

# Compilar el modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train_encoded, epochs=10, batch_size=32, validation_data=(X_test, y_test_encoded))

# Guardar el modelo entrenado
model.save('card_classifier.h5')
