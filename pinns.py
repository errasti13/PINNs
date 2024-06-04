import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


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
    def train_step(self, loss_function, x_f, t_f, u0, x0, t0):
        with tf.GradientTape() as tape:
            loss = loss_function(self.model, x_f, t_f, u0, x0, t0)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def train(self, loss_function, data, epochs=50000, print_interval=100):
        x_f, t_f, u0, x0, t0 = data
        for epoch in range(epochs):
            loss = self.train_step(loss_function, x_f, t_f, u0, x0, t0)
            if epoch % print_interval == 0:
                print(f"Epoch {epoch}: Loss = {loss.numpy()}")

    def predict(self, X):
        return self.model.predict(X)


class BurgersEquation:
    def __init__(self, nu=0.01 / np.pi):
        self.nu = nu

    def generate_data(self, x_range, t_range, N0=100, Nf=10000, sampling_method='uniform'):
        x_min, x_max = x_range
        t_min, t_max = t_range
        
        if sampling_method == 'random':
            x0 = (np.random.rand(N0, 1) * (x_max - x_min) + x_min).astype(np.float32)  # Initial condition: x in [x_min, x_max]
        elif sampling_method == 'uniform':
            x0 = np.linspace(x_min, x_max, N0)[:, None].astype(np.float32)  # Uniform sampling: x in [x_min, x_max]
        
        t0 = np.full((N0, 1), t_min, dtype=np.float32)  # Initial condition: t = t_min
        u0 = -np.sin(2 * np.pi * x0 / (x_max - x_min)).astype(np.float32)  # Initial velocity

        x_f = (np.random.rand(Nf, 1) * (x_max - x_min) + x_min).astype(np.float32)  # Collocation points: x in [x_min, x_max]
        t_f = (np.random.rand(Nf, 1) * (t_max - t_min) + t_min).astype(np.float32)  # Collocation points: t in [t_min, t_max]

        return x_f, t_f, u0, x0, t0

    def loss_function(self, model, x_f, t_f, u0, x0, t0):
        x_f = tf.convert_to_tensor(x_f, dtype=tf.float32)
        t_f = tf.convert_to_tensor(t_f, dtype=tf.float32)
        x0 = tf.convert_to_tensor(x0, dtype=tf.float32)
        t0 = tf.convert_to_tensor(t0, dtype=tf.float32)
        u0 = tf.convert_to_tensor(u0, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x_f, t_f, x0, t0])
            
            u_pred = model(tf.concat([x_f, t_f], axis=1))
            u0_pred = model(tf.concat([x0, t0], axis=1))
            
            u_x = tape.gradient(u_pred, x_f)
            u_t = tape.gradient(u_pred, t_f)
            u_xx = tape.gradient(u_x, x_f)
            
        f = u_t + u_pred * u_x - self.nu * u_xx
        u0_loss = tf.reduce_mean(tf.square(u0_pred - u0))
        f_loss = tf.reduce_mean(tf.square(f))
        
        return u0_loss + f_loss


# Visualization function
def visualize_solution(model, X_pred, T_pred, x0, u0):
    u_pred = model.predict(np.hstack((X_pred.flatten()[:, None], T_pred.flatten()[:, None])))
    U_pred = u_pred.reshape(X_pred.shape)

    # Plot the results as a contour plot
    plt.figure(figsize=(10, 6))
    plt.contourf(T_pred, X_pred, U_pred, levels=100, cmap='jet')
    plt.colorbar()
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('Predicted u(x,t)')
    plt.show()

    # 3D Surface Plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(T_pred, X_pred, U_pred, cmap='jet')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('u(x,t)')
    ax.set_title('3D Surface plot of u(x,t)')
    plt.show()

    def find_nearest(a, a0):
        diff = np.abs(a - a0)
        min_diff = diff.min()
        idx = np.argwhere(diff == min_diff)
        
        return idx

    idx = find_nearest(T_pred.flatten(), 0)

    plt.figure(figsize=(10, 6))
    plt.plot(x0, u0, 'bo', label='Initial condition')
    plt.plot(X_pred.flatten()[idx], u_pred.flatten()[idx], 'r-', label='Predicted at t=0')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend()
    plt.title('Initial Condition vs Predicted Solution at t=0')
    plt.show()

    # Plot settings
    plt.figure(figsize=(6, 4))
    
    value = 0
    idx = find_nearest(T_pred.flatten(), value)
    plt.plot(X_pred.flatten()[idx], u_pred.flatten()[idx], label=f'Predicted at t={value}')
    
    value = 0.25
    idx = find_nearest(T_pred.flatten(), value)
    plt.plot(X_pred.flatten()[idx], u_pred.flatten()[idx], label=f'Predicted at t={value}')
    
    value = 0.5
    idx = find_nearest(T_pred.flatten(), value)
    plt.plot(X_pred.flatten()[idx], u_pred.flatten()[idx], label=f'Predicted at t={value}')
    
    value = 0.75
    idx = find_nearest(T_pred.flatten(), value)
    plt.plot(X_pred.flatten()[idx], u_pred.flatten()[idx], label=f'Predicted at t={value}')
    
    value = 1
    idx = find_nearest(T_pred.flatten(), value)
    plt.plot(X_pred.flatten()[idx], u_pred.flatten()[idx], label=f'Predicted at t={value}')
         
    plt.xlabel('X')
    plt.ylabel('u')
    plt.title(f'Predicted Data')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Main function to set up and run the problem
def main():
    # Create the Burgers' equation problem
    burgers_eq = BurgersEquation()

    # Define the domain
    x_range = (-1, 1)
    t_range = (0, 1)

    # Generate data
    x_f, t_f, u0, x0, t0 = burgers_eq.generate_data(x_range, t_range, N0=100, Nf=10000, sampling_method='uniform')

    # Create the PINN model
    pinn = PINN()

    # Train the model
    pinn.train(burgers_eq.loss_function, (x_f, t_f, u0, x0, t0), epochs=10000, print_interval=100)

    # Prediction grid
    x_pred = np.linspace(x_range[0], x_range[1], 100)[:, None].astype(np.float32)
    t_pred = np.linspace(t_range[0], t_range[1], 100)[:, None].astype(np.float32)
    X_pred, T_pred = np.meshgrid(x_pred, t_pred)

    # Visualize the solution
    visualize_solution(pinn, X_pred, T_pred, x0, u0)

if __name__ == "__main__":
    main()
