from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
#RUN py -m pip install importlib_resources BEFORE RUNNING THE FILE

DATA_DIR = './tensorflow-datasets/'

train = tfds.load('mnist', split='train[:90%]', data_dir=DATA_DIR)
validation = tfds.load('mnist', split='train[-10%:]', data_dir=DATA_DIR)
test = tfds.load('mnist',split='test',data_dir=DATA_DIR)

def preprocess(sample):
  image = tf.reshape(tf.cast(sample['image'], tf.float32) / 255.0, [784])
  label = sample['label']
  return image, label

train_fit = (train.map(preprocess).shuffle(1024).batch(32))
validation_fit = (validation.map(preprocess).batch(32))
test_fit = (test.map(preprocess).batch(32))

#1 hidden layer
model1 = tf.keras.Sequential()
model1.add(
  tf.keras.layers.Dense( 64, tf.nn.relu, name='hidden')
)
model1.add(
  tf.keras.layers.Dense(10, name='final')
)

#2 hidden layers
model2 = tf.keras.Sequential()
model2.add(
  tf.keras.layers.Dense(64, tf.nn.relu, name='hidden1')
)
model2.add(
  tf.keras.layers.Dense(64, tf.nn.relu, name='hidden2')
)
model2.add(
  tf.keras.layers.Dense(10, name='final')
)


#3 hidden layers
model3 = tf.keras.Sequential()
model3.add(
  tf.keras.layers.Dense(64, tf.nn.relu, name='hidden1')
)

model3.add(
  tf.keras.layers.Dense(64, tf.nn.relu, name='hidden2')
)
model3.add(
  tf.keras.layers.Dense(64,  tf.nn.relu,  name='hidden3')
)
model3.add(
  tf.keras.layers.Dense(10,  name='final')
)

models = [model1, model2, model3]
for model in models:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

#Training The Data
history1 = model1.fit(
    train_fit,
    validation_data=validation_fit,
    epochs=20,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)]
)

history2 = model2.fit(
    train_fit,
    validation_data=validation_fit,
    epochs=20,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)]
)

history3 = model3.fit(
    train_fit,
    validation_data=validation_fit,
    epochs=20,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)]
)

#Evaluating the Data

test_loss1, test_accuracy1 = model1.evaluate(test_fit)
test_loss2, test_accuracy2 = model2.evaluate(test_fit)
test_loss3, test_accuracy3 = model3.evaluate(test_fit)

print()
print("Model 1 Accuracy:", test_accuracy1)
print("Model 2 Accuracy:", test_accuracy2)
print("Model 3 Accuracy:", test_accuracy3)

print()
print("Model 1 Loss:", test_loss1)
print("Model 2 Loss:", test_loss2)
print("Model 3 Loss:", test_loss3)

print()
print("Model 1 Epochs:", len(history1.history['loss']))
print("Model 2 Epochs:", len(history2.history['loss']))
print("Model 3 Epochs:", len(history3.history['loss']))

#Error Analysis

def error_analysis(model, dataset):
  total = np.zeros(10)
  incorrect = np.zeros(10)

  for images, labels in dataset:
    logits = model(images)
    predictions = tf.argmax(logits, axis=1)

    for label, prediction in zip(labels.numpy(), predictions.numpy()):
      total[label] += 1
        if label != prediction:
          incorrect[label] += 1
  print("Digit   Incorrect   Total   Error Rate")

  for digit in range(10):
    error_rate = incorrect[digit] / total[digit]
    print(digit, "     ", int(incorrect[digit]), "       ", int(total[digit]), "   ", error_rate)
print("")
print("Model 2 Error Analysis")
error_analysis(model2, test_fit)

#Plotting the Data

plt.plot(
    history1.history['val_accuracy'],
    label='1 hidden layer'
)
plt.plot(
    history2.history['val_accuracy'],
    label='2 hidden layers'
)
plt.plot(
    history3.history['val_accuracy'],
    label='3 hidden layers'
)
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy by Model')
plt.legend()
plt.show()