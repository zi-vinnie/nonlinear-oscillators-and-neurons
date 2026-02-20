import numpy as np
import matplotlib.pyplot as plt
from helpers import *


def fitzHughNagumoModel(
    x0: float, y0: float, dt: float, endTime: float, params: FitzHughNagumoModelParams
) -> tuple[float, float]:
    positionValues = [np.array([x0, y0])]
    # Check if the endTime is a multiple of the dt
    numSteps = int(endTime / dt)

    for i in range(numSteps):
        # Gets the dx/dt and dy/dt functions with the parameters so we can time step the model
        dx_dt = dx_dtWithParams(params.I)
        dy_dt = dy_dtWithParams(params.mu, params.a, params.b)
        # Run the euler method with the current position and the parameters
        x, y = euler_step_2d(
            positionValues[-1][0], positionValues[-1][1], dt, dx_dt, dy_dt
        )
        # Append the new position to the list of position values
        positionValues.append(np.array([x, y]))

    return np.array(positionValues)


# Main function
if __name__ == "__main__":
    # Define the parameters for the FitzHugh-Nagumo model
    IValues = np.linspace(0, 2, 11)
    mu = 10
    a = 0.8
    b = 0.7
    x0 = 2
    y0 = 1
    dt = 0.1
    endTime = 100
    # Run the FitzHugh-Nagumo model for each value of I
    for I in IValues:
        positionValues = fitzHughNagumoModel(
            x0, y0, dt, endTime, FitzHughNagumoModelParams(I, mu, a, b)
        )
        # Plot the position values on a graph
        xValues = positionValues[:, 0]
        yValues = positionValues[:, 1]
        plt.plot(xValues, yValues)
    plt.show()
