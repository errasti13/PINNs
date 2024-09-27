import tensorflow as tf

class PINN:
    def __init__(self, input_shape=(2,), layers=[20, 20, 20], activation='tanh', learning_rate=0.01, NOutputs = 1):
        self.model = self.create_model(input_shape, layers, activation, NOutputs)
        self.model.summary()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_schedule(learning_rate))
        

    def create_model(self, input_shape, layers, activation, NOutputs = 1):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
        for units in layers:
            model.add(tf.keras.layers.Dense(units, activation=activation))
        model.add(tf.keras.layers.Dense(NOutputs))  # Output layer
        return model

    def learning_rate_schedule(self, initial_learning_rate):
        return tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.9
        )

    @tf.function
    def train_step(self, loss_function, data):
        with tf.GradientTape() as tape:
            loss = loss_function(self.model, data)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def train(self, loss_function, data, epochs=50000, print_interval=100):
        for epoch in range(epochs):
            loss = self.train_step(loss_function, data)
            if (epoch + 1) % print_interval == 0:
                print(f"Epoch {epoch + 1}: Loss = {loss.numpy()}")

    def predict(self, X):
        return self.model.predict(X)
