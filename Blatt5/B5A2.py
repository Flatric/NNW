from tensorflow import keras
import numpy as np

#daten laden
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#lables
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#farben weg
train_images = train_images / 255.0
test_images = test_images / 255.0 



model = keras.models.Sequential()

model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Reshape(input_shape=(28, 28), target_shape = (28, 28)+(1,)))

# Convolutional Layer mit 8 3x3 groÿen Kernels, ReLU Aktivierung, Padding same
model.add(keras.layers.Conv2D(filters = 8, kernel_size = (3,3), activation="relu", strides=(1, 1), padding='same', name=None))

#Max-Pooling Layer mit Gröÿe 2x2 und Stride 2.
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

#Convolutional Layer mit 16 3x3 groÿen Kernels, ReLU Aktivierung, Padding same
model.add(keras.layers.Conv2D(filters = 16, kernel_size = (3,3), activation="relu", strides=(1, 1), padding='same', name=None))

#Max-Pooling Layer mit Gröÿe 2x2 und Stride 2.
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

#Fully-connected Layer mit 20 Neuronen, ReLU Aktivierung
model.add(keras.layers.Dense(units=20, activation="relu", name="fully_connected"))

#Ausgabe-Layer
#ka welche parameter, klau ich einfach beim colab
model.add(keras.layers.Dense(units=10, name="output_layer"))


#compilen, zu Statt sparse nur Categorical weil nicht one hot
model.compile(optimizer='adam',
              loss=keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


#training 
model.fit(train_images, train_labels, epochs=10)


#evaluieren
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)
model.summary()


# test prediction
probability_model = keras.Sequential([model, keras.layers.Softmax()])

predictions = probability_model.predict(test_images)

print("prediction: ", predictions[0])

print("in schön:", np.argmax(predictions[0]))

print("testlable: ", test_labels[0])