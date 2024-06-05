import tensorflow as tf

class PINN:
    def __init__(self, input_shape=(2,), layers=[20, 20, 20], activation='tanh', learning_rate=0.01):
        self.model = self.create_model(input_shape, layers, activation)
        self.model.summary()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_schedule(learning_rate))

    def create_model(self, input_shape, layers, activation):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
        for units in layers:
            model.add(tf.keras.layers.Dense(units, activation=activation))
        model.add(tf.keras.layers.Dense(1))  # Output layer
        return model

    def learning_rate_schedule(self, initial_learning_rate):
        return tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.9
        )

    @tf.function
    def train_step(self, loss_function, x_f, t_f, u0, x0, t0, xBc0, tBc0, uBc0, xBc1, tBc1, uBc1):
        with tf.GradientTape() as tape:
            loss = loss_function(self.model, x_f, t_f, u0, x0, t0, xBc0, tBc0, uBc0, xBc1, tBc1, uBc1)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def train(self, loss_function, data, epochs=50000, print_interval=100):
        x_f, t_f, u0, x0, t0, xBc0, tBc0, uBc0, xBc1, tBc1, uBc1 = data
        for epoch in range(epochs):
            loss = self.train_step(loss_function, x_f, t_f, u0, x0, t0, xBc0, tBc0, uBc0, xBc1, tBc1, uBc1)
            if epoch % print_interval == 0:
                print(f"Epoch {epoch}: Loss = {loss.numpy()}")

    def predict(self, X):
        return self.model.predict(X)
