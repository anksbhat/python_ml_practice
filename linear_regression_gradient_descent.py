# linear regression using gradient descent 

# PROBLEM STATEMENT 

# Write a Python function that performs linear regression using gradient descent. 
# The function should take NumPy arrays X (features with a column of ones for the intercept) and y (target) as input, along with learning rate alpha and the number of iterations, and 
# return the coefficients of the linear regression model as a NumPy array. Round your answer to four decimal places. -0.0 is a valid result for rounding a very small number.

import numpy as np 

def linear_regression(x, y, alpha, num_iterations)-> np.ndarray:
    m = len(y)
    theta = np.zeros(x.shape[1])

    for i in range(num_iterations):
         # Compute predictions
        predictions = x.dot(theta)

        # Compute the error (difference between predictions and actual values)
        errors = predictions - y

        # Compute the gradient of the cost function with respect to each weight
        gradient = (1/m)*x.t.dot(errors)

        # Update the weights by moving in the opposite direction of the gradient

        theta-= alpha*gradient 

    theta = np.round(theta, 4)

    return theta