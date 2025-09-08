# solve a system of linear equations using CRAMER's rule , Help me Implement this function with python,
# If there is a system of linear equations Ax = b, A being the square coefficient matrix and a constant vector b, 
# return a list of solution vector x. If the system has no solution, that is if the determinant of A is 0 we need to return -1

import numpy as np 

def cramers_rule(A,b):
    det_A = np.linalg.det(A)
    if det_A < 1e-9:
        return -1
    else:
        n = A.shape[0]
        x = [] 
        for i in range(n):
            A_i = A.copy()
            A_i[:,i] = b
            det_A_i = np.linalg.det(A_i)
            x_i = det_A_i / det_A
            x.append(x_i)
        return x

if __name__ == "__main__":
    print("--- Testing Cramer's Rule Solver ---")

    SOLUTION_LABEL = "Solution x:"

    # Example 1: A simple 2x2 system
    # x + y = 3
    # 2x - y = 0
    A1 = np.array([[1, 1], [2, -1]])
    b1 = np.array([3, 0])
    print("\nSystem 1:")
    print("A:\n", A1)
    print("b:", b1)
    solution1 = cramers_rule(A1, b1)
    if solution1 != -1:
        print(SOLUTION_LABEL, [round(val, 6) for val in solution1]) # Round for cleaner output
        # Expected: x=1, y=2
        # Verify: 1+2=3, 2*1-2=0
    else:
        print("No unique solution.")

    # Example 2: A 3x3 system
    # x + 2y + 3z = 6
    # 2x - y + z = 2
    # 3x + y - 2z = 1
    A2 = np.array([[1, 2, 3], [2, -1, 1], [3, 1, -2]])
    b2 = np.array([6, 2, 1])
    print("\nSystem 2:")
    print("A:\n", A2)
    print("b:", b2)
    solution2 = cramers_rule(A2, b2)
    if solution2 != -1:
        print(SOLUTION_LABEL, [round(val, 6) for val in solution2])
        # Expected: x=1, y=1, z=1
    else:
        print("No unique solution.")

    # Example 3: Singular matrix (no unique solution)
    # x + y = 2
    # 2x + 2y = 4 (or any other value)
    A3 = np.array([[1, 1], [2, 2]])
    b3 = np.array([2, 4]) # Or b3 = np.array([2, 5]) for no solution
    print("\nSystem 3 (Singular):")
    print("A:\n", A3)
    print("b:", b3)
    solution3 = cramers_rule(A3, b3)
    if solution3 != -1:
        print(SOLUTION_LABEL, [round(val, 6) for val in solution3])
    else:
        print("No unique solution (as expected).")
    