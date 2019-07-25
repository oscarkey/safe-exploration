import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from scipy.integrate import ode
from mpl_toolkits.mplot3d import Axes3D

n = 3
l = 0.5
m = 0.15
g = 9.82

g_vec = np.array([0., 0., -g])
dt = 0.1


def dynamics(t, state, action):
    assert state.shape[0] == (n - 1) * 2
    assert action.shape[0] == (n - 1)
    velocity = state[:(n - 1)]
    position = state[(n - 1):]

    gravity_proj = np.zeros_like(position)
    gravity_proj[0] = g / l * np.sin(position[0])

    inertia = m * l ** 2

    dvelocity = gravity_proj + action / inertia  # - b / inertia * state[0]
    dposition = velocity

    return np.concatenate((dvelocity, dposition))


theta1 = np.array(1.0)
theta2 = np.array(0.)
theta1_dot = np.array(0.5)
theta2_dot = np.array(0.2)
state = np.array([theta1_dot, theta2_dot, theta1, theta2])

solver = ode(dynamics)
solver.set_initial_value(state, t=0.0)

for t in range(100):
    solver.set_f_params(np.array([0., 0.]))
    next_state = solver.integrate(solver.t + dt)

    theta1 = next_state[2]
    theta2 = next_state[3]
    theta = theta1
    phi = theta2

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.azim = 90
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.plot([0, 1], [0, 0], [0, 0], color='grey')
    ax.plot([0, 0], [0, 1], [0, 0], color='grey')
    ax.plot([0, 0], [0, 0], [0, 1], color='grey')
    ax.plot([0, x], [0, y], [0, z])
    ax.scatter([x], [y], [z])
    plt.show()
