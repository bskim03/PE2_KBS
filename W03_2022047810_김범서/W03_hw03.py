# 2022047810 김범서

import numpy as np

# Input N, M, X, Y values
N = int(input("N: "))
M = np.array(list(map(int, input("M: ").split(" ", maxsplit=N))))
X, Y = map(int, input("X, Y: ").split(" ", maxsplit=2))

# Create teachers array, initialized to 0
teachers = np.zeros(N)

for i in range(N):
    # Calculate the number of teachers for each playground
    # Compare the case with and without homeroom teacher
    if X > M[i]:
        case1 = 1
    else:
        case1 = 1 + np.ceil((M[i] - X) / Y)
    case2 = np.ceil(M[i] / Y)
    teachers[i] = np.array([case1, case2]).min()

# Print the results
print(int(teachers.sum()))
