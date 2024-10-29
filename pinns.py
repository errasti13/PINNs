import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

class PINN:
    def __init__(self, input_shape=2, output_shape=1, layers=[20, 20, 20], activation='tanh', learning_rate=0.01):
        self.model = self.create_model(input_shape, output_shape, layers, activation)
        self.model.summary()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_schedule(learning_rate))
        

    def create_model(self, input_shape,  output_shape, layers, activation):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
        for units in layers:
            model.add(tf.keras.layers.Dense(units, activation=activation))
        model.add(tf.keras.layers.Dense(output_shape))  # Output layer
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



    def train(self, loss_function, data, epochs=50000, print_interval=100, autosave_interval = 10000):
        loss_history = []
        epoch_history = []

        # Set up the plot
        plt.ion()  # Turn on interactive mode for live updates
        fig, ax = plt.subplots()
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_yscale('log')  # Set y-axis to logarithmic scale

        # Ensure that the ticks are in decimal format (e.g., 1.0, 0.1, etc.)
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(ScalarFormatter())
        ax.yaxis.get_major_formatter().set_useOffset(False)
        ax.yaxis.get_major_formatter().set_scientific(False)

        line, = ax.semilogy([], [], label='Training Loss')
        plt.legend()

        for epoch in range(epochs):
            loss = self.train_step(loss_function, data)

            # Update loss and epoch history only at intervals
            if (epoch + 1) % print_interval == 0:
                loss_history.append(loss.numpy())
                epoch_history.append(epoch + 1)

                # Update plot
                line.set_xdata(epoch_history)
                line.set_ydata(loss_history)
                ax.relim()  # Recompute the data limits
                ax.autoscale_view()  # Rescale the view to fit the new data

                plt.draw()
                plt.pause(0.001)  # Pause to update the figure

                print(f"Epoch {epoch + 1}: Loss = {loss.numpy()}")

            if (epoch + 1) % autosave_interval == 0:
                self.model.save(f'trainedModels/{self.eq}.tf')

        plt.ioff()  # Turn off interactive mode
        plt.close()

    def predict(self, X):
        return self.model.predict(X)
