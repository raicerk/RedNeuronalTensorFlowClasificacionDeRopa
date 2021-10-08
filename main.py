import tensorflow as tf
import tensorflow_datasets as tfds
import math
import numpy as np

datos, metadatos = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
datos_entrenamiento, datos_pruebas = datos['train'], datos['test']
nombres_clases = metadatos.features['label'].names

# Normalizar los datos (Pasar de 0-255 a 0-1)
def normalizar(imagenes, etiquetas):
  imagenes = tf.cast(imagenes, tf.float32)
  imagenes /= 255 #aqui lo pasa de 0-255 a 0-1
  return imagenes, etiquetas

# Normalizar los datos de entrenamiento y pruebas con la funcion que hicimos
datos_entrenamiento = datos_entrenamiento.map(normalizar)
datos_pruebas = datos_pruebas.map(normalizar)

# Agregar a cache (usar memoria en lugar de disco, entrenamiento más rápido)
datos_entrenamiento = datos_entrenamiento.cache()
datos_pruebas = datos_pruebas.cache()

for imagen, etiqueta in datos_entrenamiento.take(1):
  break
imagen = imagen.numpy().reshape((28,28)) #Redimensionar, cosas de tensores

# Creación de modelo
modelo = tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape=(28,28, 1)), #1 - blanco y negro
  tf.keras.layers.Dense(50, activation=tf.nn.relu), # Capa oculta de 50 neuronas y función de activación relu para que retorne solo números positivos
  tf.keras.layers.Dense(50, activation=tf.nn.relu), # Capa oculta de 50 neuronas y función de activación relu para que retorne solo números positivos
  tf.keras.layers.Dense(10, activation=tf.nn.softmax) # Capa de salida de 10 neuronas y función de activación para solo retornar el valor más alto correspondiente
])

# Compilar el modelo
modelo.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

num_ej_entrenamiento = metadatos.splits['train'].num_examples
num_ej_pruebas = metadatos.splits['test'].num_examples

print(num_ej_entrenamiento) # Cantidad de datos para entrenar
print(num_ej_pruebas) # Cantidad de datos para testear

# Lotes de procesamiento, se usa procesamiento por lotes para no colapsar y tratar todos los datos de manera secuencial
TAMANO_LOTE = 32
datos_entrenamiento = datos_entrenamiento.repeat().shuffle(num_ej_entrenamiento).batch(TAMANO_LOTE)
datos_pruebas = datos_pruebas.batch(TAMANO_LOTE)


#Entrenar
historial = modelo.fit(datos_entrenamiento, epochs=5, steps_per_epoch=math.ceil(num_ej_entrenamiento/TAMANO_LOTE))


# Tomar cualquier indice del set de pruebas para ver su predicción
for imagenes_prueba, etiquetas_prueba in datos_pruebas.take(1):
  imagenes_prueba = imagenes_prueba.numpy()
  etiquetas_prueba = etiquetas_prueba.numpy()
  predicciones = modelo.predict(imagenes_prueba)

imagen = imagenes_prueba[11]
imagen = np.array([imagen])
prediccion = modelo.predict(imagen)
print("Predicción: " + nombres_clases[np.argmax(prediccion[0])])