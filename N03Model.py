import tensorflow as tf

class Dense(tf.Module):
    def __init__(self, output_size, activation=tf.nn.relu, name=None):
        super(Dense, self).__init__(name=name)
        self.output_size = output_size
        self.activation = activation
        self.is_built = False

    @tf.function
    def __call__(self, x):
        if not self.is_built:
            self.w = tf.Variable(
                tf.random.normal([x.shape[-1], self.output_size]),
                name='w'
            )
            
            self.b = tf.Variable(
                tf.zeros([self.output_size]),
                name='b'
            )
            self.is_built = True
        return self.activation(tf.matmul(x, self.w) + self.b)

dense_layer = Dense(5)
dense_layer(tf.ones([3, 3]))
print("Saving model...")

tf.saved_model.save(
    dense_layer,
    './saved_model'
)

print("Model saved.")