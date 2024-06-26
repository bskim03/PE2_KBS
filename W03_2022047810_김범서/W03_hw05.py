# 2022047810 김범서

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 2, 0.2)

# Display graph by -
# r--: Red dotted line,
# bo: Blue dot,
# g-.: green line with dots
plt.plot(x, x, 'r--', x, x ** 2, 'bo', x, x ** 3, 'g-.')

plt.show()
