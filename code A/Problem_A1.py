# =================================================================================
# PROBLEM A1
#
# Given two arrays, train a neural network model to match the X to the Y.
# Predict the model with new values of X [-2.0, 10.0]
# We provide the model prediction, do not change the code.
#
# The test infrastructure expects a trained model that accepts
# an input shape of [1].
# Do not use lambda layers in your model.
#
# Please be aware that this is a linear model.
# We will test your model with values in a range as defined in the array to make sure your model is linear.
#
# Desired loss (MSE) < 1e-4
# =================================================================================
<<<<<<< HEAD
#oke
=======

>>>>>>> 4f175b36e4ac0618472acd7cf57cc92d01582a99

import numpy as np
import tensorflow as tf
from tensorflow import keras


def solution_A1():
    # DO NOT CHANGE THIS CODE
    x = np.array([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0,
                 2.0, 3.0, 4.0, 5.0], dtype=float)
    y = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
                 12.0, 13.0, 14.0, ], dtype=float)

    class Callback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs['loss'] < 1e-4:
                print("\nHentikan Training.")
                self.model.stop_training = True

    callback = Callback()
    models = tf.keras.models.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    models.compile(optimizer='sgd',
                  loss='mean_squared_error')
<<<<<<< HEAD
    models.fit(x,y,epochs=1010, callbacks=[callback])

    print(models.predict([-2.00, 10.0]))
=======
    models.fit(x,y,epochs=1000, callbacks=[callback])

    print(models.predict([-2.0, 10.0]))
>>>>>>> 4f175b36e4ac0618472acd7cf57cc92d01582a99
    return models


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_A1()
    model.save("model_A1.h5")
