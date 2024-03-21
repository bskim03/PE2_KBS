# 2022047810 김범서

import numpy as np

# Input matrix size
n = int(input("n = "))

# Create matrices A and B
A = np.zeros(shape=(n, n))
B = np.zeros(shape=n)

# Input values for each column of matrix A
print("A를 입력", end="\n")
for i in range(n):
    while True:
        if len(a_input := list(map(int, input("").split(" ")))) == n:
            A[i] = a_input
            break
        else:
            print("error")
            continue

# Input values for matrix B
print("B를 입력", end="\n")
while True:
    if len(b_input := list(map(int, input("").split(" ")))) == n:
        B = b_input
        break
    else:
        print("error")
        continue

# Solve the system of linear equations
X = np.linalg.solve(A, B)

# Print the solution
print("Solution:", X)
