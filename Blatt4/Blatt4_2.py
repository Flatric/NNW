from tensorflow import keras
import numpy as np
from nnwplot import plotTwoFeatures
from matplotlib import pyplot as plt

def plot_loss(loss,val_loss):
    plt.figure()
    plt.plot(loss); plt.plot(val_loss)
    plt.title('model loss'); plt.ylabel('loss'); plt.xlabel('epoch')
    plt.legend(['loss', 'accuracy'], loc='upper right')
    plt.show()

def plotty(epoch,logs):
    if (epoch%100==0):
        plotTwoFeatures(X,T,model.predict)


input = keras.Input(2,) #tupel


a = keras.layers.Dense(units=5, activation="tanh", input_shape=(2,), name="hidden")(input)
out = keras.layers.Dense(units=3, activation="softmax", name="output_layer")(a)

model = keras.models.Model(input, out)

model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])

iris = np.loadtxt(fname="iris.csv", delimiter=",")

X = iris[:, :-1]
T = iris[:, -1]
X = X[:, 2:]

plot_every_ten = keras.callbacks.LambdaCallback(on_epoch_end=plotty)

myhistory = model.fit(x=X,y=keras.utils.to_categorical(T),epochs=500, callbacks=[plot_every_ten])

plot_loss(myhistory.history['loss'], myhistory.history['accuracy'])

model.summary()
keras.utils.plot_model(model, to_file="model.png")


plotTwoFeatures(X,T,model.predict)

