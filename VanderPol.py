import numpy as np 
import matplotlib.pyplot as plt
import os
from helpers import *

def VanderPolModel(
    x0: float, y0: float, dt: float, endTime: float, params: ModelParams
) -> tuple[float, float]:
    positionValues = [np.array([x0, y0])]
    positionValues2 = [np.array([x0, y0])]
    numSteps = int(endTime / dt)

    for i in range(numSteps):
        # Gets the dx/dt and dy/dt functions with the parameters so we can time step the model
        # The euler method requires a dx_dt and dy_dt function with two parameters (x, y)
        # So by using dx_dtWithParams and dy_dtWithParams, we can pass the parameters (I, mu, a, b) to the functions
        dx_dt = dx_dtWithParams(params.I)
        dy_dt = dy_dtWithParams(params.mu, params.a, params.b)
        # Run the euler method with the current position and the parameters
        x, y = euler_step_2d(
            # positionValues[-1] indicates the last position, 
            # 0 indicates the x value 
            # 1 indicates the y value
            positionValues[-1][0], positionValues[-1][1], dt, dx_dt, dy_dt
        )
        # Append the new position to the list of position values
        positionValues.append(np.array([x, y]))

        x2, y2 = midpoint_step_2d( 
            # positionValues[-1] indicates the last position, 
            # 0 indicates the x values 
            # 1 indicates the y values
            positionValues2[-1][0], positionValues2[-1][1], dt, dx_dt, dy_dt
        )
        # Append the new position to the list of positive values
        positionValues2.append(np.array([x2,y2]))

    # Return the position values as a numpy array
    return np.array(positionValues), np.array(positionValues2)


# Main function
# This code here checks if the file is being run directly and not imported as a module
if __name__ == "__main__":
    # Define the parameters for the FitzHugh-Nagumo model
    IValues = np.linspace(0, 2, 11)
    mu = 10
    a = 0.8
    b = 0.7
    x0 = 2
    y0 = 0
    dt = 0.1
    endTime = 33

    fig, axs = plt.subplots()
    # Run the Van der Pol model for each value of I
    for I in IValues:
        # Create the parameters for the FitzHugh-Nagumo model
        params = ModelParams(0, mu, 0, 0)
        # Run the FitzHugh-Nagumo model for the current value of I
        positionValues, positionValues2 = VanderPolModel(
            x0, y0, dt, endTime, params
        )
        # Plot the position values on a graph
        xValues = positionValues[:, 0]
        yValues = positionValues[:, 1]
        xValues2 = positionValues2[:, 0] 
        yValues2 = positionValues2[:, 1]
        #axs.plot(xValues, yValues)
        axs.plot(xValues2, yValues2, linestyle = "--")
    # Save the figure
    # Check if the directory exists, if not create it
    if not os.path.exists("VanderPol"):
        os.makedirs("VanderPol")

    fig.savefig("VanderPol/vanderpol.png", dpi=300)

    