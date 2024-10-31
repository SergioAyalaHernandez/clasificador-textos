import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D, Conv1D, MaxPooling1D, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import matplotlib.pyplot as plt
import pickle

# Cargar datos
df = pd.read_csv('casos.csv', encoding='ISO-8859-1')
print("Datos cargados:")
print(df.head())  # Imprime las primeras filas del DataFrame
df['targets'] = df['tipo'].astype("category").cat.codes

# Dividir datos
df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)

# Preparar datos de texto
MAX_VOCAB_SIZE = 20000
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(df_train['detalle'])
sequences_train = tokenizer.texts_to_sequences(df_train['detalle'])
sequences_test = tokenizer.texts_to_sequences(df_test['detalle'])

word2idx = tokenizer.word_index
V = len(word2idx)
print(f'Encontrado {V} tokens')

data_train = pad_sequences(sequences_train)
data_test = pad_sequences(sequences_test, maxlen=data_train.shape[1])

# Convertir etiquetas a arrays NumPy
y_train = np.array(df_train['targets'])
y_test = np.array(df_test['targets'])

# Definir el modelo
D = 50
i = Input(shape=(data_train.shape[1],))
x = Embedding(V + 1, D)(i)
x = Conv1D(32, 3, activation='relu')(x)
x = MaxPooling1D(3)(x)
x = Conv1D(64, 3, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(len(df['targets'].unique()), activation='softmax')(x)
model = Model(i, x)

model.compile(
    loss=SparseCategoricalCrossentropy(from_logits=False),
    optimizer='adam',
    metrics=['accuracy']
)

# Calcular ponderaciones de clases
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

print("Ponderaciones de clases:")
for etiqueta, peso in class_weight_dict.items():
    print(f"Etiqueta {etiqueta}: Peso {peso}")

# Entrenar el modelo con ponderaciones de clases
print('Entrenando el modelo')
r = model.fit(
    data_train,
    y_train,
    epochs=50,
    validation_data=(data_test, y_test),
    class_weight=class_weight_dict
)

# Imprimir resultados de entrenamiento
print('Resultados de entrenamiento:')
print('Historial de entrenamiento:')
print(r.history)

# Guardar el modelo
model.save('modelo.keras')
model.save_weights('pesos.weights.h5')

# Guardar el Tokenizer y el tamaño máximo de secuencia
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

with open('max_length.pkl', 'wb') as f:
    pickle.dump(data_train.shape[1], f)

# Graficar las métricas de entrenamiento
plt.figure(figsize=(12, 5))

# Gráfico de la pérdida
plt.subplot(1, 2, 1)
plt.plot(r.history['loss'], label='Train Loss')
plt.plot(r.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()

# Gráfico de la precisión
plt.subplot(1, 2, 2)
plt.plot(r.history['accuracy'], label='Train Accuracy')
plt.plot(r.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

plt.show()

word_freq = tokenizer.word_counts
sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

# Convertir a DataFrame para mejor visualización
df_word_freq = pd.DataFrame(sorted_word_freq, columns=['Word', 'Frequency'])

# Mostrar las 10 palabras más frecuentes
print("Top 10 palabras más frecuentes:")
print(df_word_freq.head(10))

# Graficar las palabras más frecuentes
plt.figure(figsize=(12, 8))
plt.bar(df_word_freq['Word'].head(20), df_word_freq['Frequency'].head(20))
plt.xlabel('Palabra')
plt.ylabel('Frecuencia')
plt.title('Top 20 palabras más frecuentes')
plt.xticks(rotation=90)

plt.show()

# Guardar ponderaciones de clases
with open('class_weights.pkl', 'wb') as f:
    pickle.dump(class_weight_dict, f)


# Evaluar el modelo en el conjunto de prueba
loss, accuracy = model.evaluate(data_test, y_test)
print(f"Pérdida en el conjunto de prueba: {loss}")
print(f"Precisión en el conjunto de prueba: {accuracy}")

# Obtener y mostrar las predicciones
predictions = model.predict(data_test)
predicted_classes = np.argmax(predictions, axis=1)

print("Clasificaciones reales vs predicciones:")
print(list(zip(y_test, predicted_classes)))


print("Archivos guardados exitosamente.")
