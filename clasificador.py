import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Cargar el modelo
modelo_cargado = load_model('modelo.keras')

# Cargar el Tokenizer y el tamaño máximo de secuencia
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('max_length.pkl', 'rb') as f:
    max_length = pickle.load(f)

# Cargar etiquetas originales
df = pd.read_csv('casos.csv', encoding='ISO-8859-1')

# Cargar ponderaciones de clases
with open('class_weights.pkl', 'rb') as f:
    class_weights = pickle.load(f)

# Obtener los nombres de las etiquetas
etiquetas = df['tipo'].astype("category").cat.categories
num_clases = len(etiquetas)
print(f"Número de etiquetas: {num_clases}")

# Mostrar los pesos de clases con nombres de etiquetas
print("Pesos de clases:")
for idx, peso in class_weights.items():
    print(f"Etiqueta '{etiquetas[idx]}': Peso {peso}")

# Función para preprocesar un nuevo texto
def preprocess_text(text, tokenizer, max_length):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length)
    return padded_sequence

# Función para predecir la etiqueta del texto
def predecir_etiqueta(texto):
    # Preprocesar el nuevo texto
    data_nuevo_texto = preprocess_text(texto, tokenizer, max_length)
    
    # Realizar la predicción
    prediccion = modelo_cargado.predict(data_nuevo_texto)
    etiqueta_predicha = np.argmax(prediccion, axis=-1)
    
    # Convertir el código de la etiqueta a la etiqueta original
    etiqueta_original = etiquetas[etiqueta_predicha[0]]
    return etiqueta_original

# Ingresar un nuevo texto y predecir su etiqueta

etiququetas = df['tipo'].astype("category").cat.categories
print("Etiquetas:", list(etiququetas))




nuevo_texto = "Cambio de jornada solicitado en Derecho Labora"
nuevo_texto2 = "Proceso de certificación en Especialización en Ciencias Forenses"
nuevo_texto3 = "Proceso de inscripción en Literatura Latinoamericana"
nuevo_texto4 = "certificacion"
etiqueta = predecir_etiqueta(nuevo_texto)
etiqueta2 = predecir_etiqueta(nuevo_texto2)
etiqueta3 = predecir_etiqueta(nuevo_texto3)
etiqueta4 = predecir_etiqueta(nuevo_texto4)



print(f"Texto: {nuevo_texto}")
print(f"Etiqueta predicha: {etiqueta}")
print('---------------------------------------------')
print(f"Texto: {nuevo_texto2}")
print(f"Etiqueta predicha: {etiqueta2}")
print('---------------------------------------------')
print(f"Texto: {nuevo_texto3}")
print(f"Etiqueta predicha: {etiqueta3}")
print('---------------------------------------------')
print(f"Texto: {nuevo_texto4}")
print(f"Etiqueta predicha: {etiqueta4}")
