import numpy as np 
import matplotlib.pyplot as plt
import os
from helpers import *
from VanderPol import * 

# Defining the parameters
mu = 10
params = ModelParams(0, mu, 0, 0)

eulerPositionsExplore, midpointPositionsExplore = VanderPolModel(x0, y0, endTime / numSteps, numSteps, params)
xValuesExplore = midpointPositionsExplore[:, 0]
yValuesExplore = midpointPositionsExplore[:, 1]

