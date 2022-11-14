from tensorflow import keras
import numpy as np
from nnwplot import plotTwoFeatures

model = keras.models.Sequential()

model.add(keras.layers.Dense(units=5, activation="tanh", input_shape=(2,), name="hidden1"))
model.add(keras.layers.Dense(units=3, activation="softmax", name="output_layer"))

model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])

iris = np.loadtxt(fname="iris.csv", delimiter=",")

X = iris[:, :-1]
T = iris[:, -1]
X = X[:, 2:]
print(f"x shape {X.shape},  T shape {T.shape}")
model.fit(x=X,y=keras.utils.to_categorical(T),epochs=1000)
print(model.summary())
keras.utils.plot_model(model, to_file="model.png")
plotTwoFeatures(X,T,model.predict)

