# ============================================
# TRABAJO PRÁCTICO INTEGRADOR – MÓDULO 3
# Detección de Fraude Bancario
# Autor: Julio Arturo Rodríguez
# ============================================

# --------- LIBRERÍAS ---------
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential

from collections import Counter
from imblearn.over_sampling import RandomOverSampler

import gdown


# --------- DESCARGA DEL DATASET ---------
url = 'https://drive.google.com/uc?id=1hga3zUzjhYqmR_aCpSIhYEktqClCIaV8'
output = 'basededatos.csv'
gdown.download(url, output, quiet=False)

# --------- CARGA DE DATOS ---------
data = pd.read_csv('basededatos.csv')

print("Primeras filas del dataset:")
print(data.head())

print("\nDistribución original de la clase:")
print(data['Class'].value_counts())


# --------- SEPARAR FEATURES Y ETIQUETA ---------
X = data.drop('Class', axis=1)
y = data['Class']


# --------- ANALIZAR DESBALANCE ---------
plt.figure(figsize=(8, 6))
data['Class'].value_counts().plot(kind='bar', color=['blue', 'red'])
plt.title('Desbalance de Clases')
plt.xlabel('Clase')
plt.ylabel('Cantidad de Muestras')
plt.xticks([0, 1], ['Normal', 'Fraude'])
plt.show()


# --------- OVERSAMPLING ---------
print("\nDistribución antes del oversampling:", Counter(y))

oversampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X, y)

print("Distribución después del oversampling:", Counter(y_resampled))

plt.figure(figsize=(8, 6))
plt.bar(Counter(y_resampled).keys(), Counter(y_resampled).values(),
        color=['blue', 'red'])
plt.title('Clases después del Oversampling')
plt.xlabel('Clase')
plt.ylabel('Cantidad de Muestras')
plt.xticks([0, 1], ['Normal', 'Fraude'])
plt.show()


# --------- TRAIN / TEST SPLIT ---------
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)


# --------- ESCALADO ---------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# --------- MAPA DE CALOR ---------
correlation_matrix = data.corr()

plt.figure(figsize=(25, 13))
sns.heatmap(correlation_matrix, cmap="RdYlGn")
plt.show()

plt.figure(figsize=(25, 13))
sns.heatmap(correlation_matrix, annot=True, cmap="RdYlGn")
plt.show()


# --------- RED NEURONAL ---------
model = Sequential()
model.add(Dense(units=128, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')

history = model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)


# --------- PREDICCIÓN Y MATRIZ DE CONFUSIÓN ---------
y_probs = model.predict(X_test)
y_pred = np.round(y_probs)

conf_matrix = confusion_matrix(y_test, y_pred)

class_names = ['No', 'Yes']
conf_df = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)

print("\nMatriz de Confusión:")
print(conf_df)

TN, FP, FN, TP = conf_matrix.ravel()
accuracy = (TP + TN) / (TP + TN + FP + FN)

print("\nAccuracy:", accuracy)


# --------- MATRIZ DE CONFUSIÓN VISUAL ---------
plt.figure(figsize=(6, 5))
sns.heatmap(conf_df, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión')
plt.ylabel('Real')
plt.xlabel('Predicción')
plt.show()