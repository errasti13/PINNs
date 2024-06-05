import matplotlib.pyplot as plt
import numpy as np

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

    timeRange = T_pred.max() - T_pred.min()
    
    value = timeRange * 0
    idx = find_nearest(T_pred.flatten(), value)
    plt.plot(X_pred.flatten()[idx], u_pred.flatten()[idx], label=f'Predicted at t={value}')
    
    value = timeRange * 0.25
    idx = find_nearest(T_pred.flatten(), value)
    plt.plot(X_pred.flatten()[idx], u_pred.flatten()[idx], label=f'Predicted at t={value}')
    
    value = timeRange * 0.5
    idx = find_nearest(T_pred.flatten(), value)
    plt.plot(X_pred.flatten()[idx], u_pred.flatten()[idx], label=f'Predicted at t={value}')
    
    value = timeRange * 0.75
    idx = find_nearest(T_pred.flatten(), value)
    plt.plot(X_pred.flatten()[idx], u_pred.flatten()[idx], label=f'Predicted at t={value}')
    
    value = timeRange * 1
    idx = find_nearest(T_pred.flatten(), value)
    plt.plot(X_pred.flatten()[idx], u_pred.flatten()[idx], label=f'Predicted at t={value}')
         
    plt.xlabel('X')
    plt.ylabel('u')
    plt.title(f'Predicted Data')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
