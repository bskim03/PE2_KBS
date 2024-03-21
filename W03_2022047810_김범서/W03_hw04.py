# 2022047810 김범서

import matplotlib.pyplot as plt
import numpy as np

# Create an x array that divides [-10, 10] into 100 values
x = np.linspace(-10, 10, 100)
y = x ** 3

# Plot x, y
plt.plot(x, y)

# Change x-axis scale to symmetric log
plt.xscale('symlog')

# Output the graph
plt.show()
