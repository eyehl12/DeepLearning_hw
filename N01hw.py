import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def optimize(learning_rate, num_iterations):
  x = tf.Variable(2.0)
  y = tf.Variable(2.0)

  optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

  losses = []
  x_values = []
  y_values = []

  for i in range(num_iterations):
    with tf.GradientTape() as tape:
      loss = x**2 + y**2 - x

    gradients = tape.gradient(loss, [x, y])
    optimizer.apply_gradients(zip(gradients, [x, y]))

    losses.append(loss.numpy())
    x_values.append(x.numpy())
    y_values.append(y.numpy())

  return x.numpy(), y.numpy(), losses, x_values, y_values

learning_rates = [0.01, 0.1, 0.5, 0.8, 1.0]
iterations_list = [10, 50, 100]

for lr in learning_rates:
  for num_iterations in iterations_list:
    x, y, losses, x_values, y_values = optimize(
      lr, num_iterations
    )
    print(
      "Learning rate:", lr,
      "Iterations:", num_iterations,
      "X:", x,
      "y:", y,
      "Final loss:", losses[-1]
    )

x, y, losses, x_values, y_values = optimize(float(input("Learning Rate (0-1): ")), 50)

plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("Function value")
plt.title("Loss vs. Iteration")
plt.show()
