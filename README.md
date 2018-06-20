# Logistic-Regression-on-Iris-Data-Set-Using-the-Artificial-Neural-Network
UCLA Master of Applied Economics Econ 413 Assignment

In this assignment you will use Determine Cost Function for the Artificial Neural Network for Iris Train and Test  assignment as the base and implement Logistic Regression. As this is simply a single layer network, you will modify Layer 1 to become unity layer. Remember, Layer 0 of the network is already unity (with W0 is a unity matrix, B0 is a zero vector, and f 0 is unit activation).
With the first two layers as unit layers, X2 will become simply as the input of the Logistic Regression system. However you must transition the input through first two layers (maintain the original network code). In other words your iteration loop still will consists of the layer equations. See below pseudocode:

    X = training data read from the file
    for h in range(g):
        X0 = X         # Assign input vector to the input of Layer 0 (unit activation)       
        G0 = W0 X0 + B0 # Determine G0   
        H0 = f0(G0)    # Apply activation function of Layer 0
        X1 = H0
        G1 = W1 X1 + B1 # Determine G1   
        H1 = f1(G1)    # Apply activation function of Layer 1 (unit activation)
        X2 = H1
        G2 = W2 X2 + B2 # Determine G2   
        H2 = f2(G2)    # Apply activation function of Layer 2 (sigmoid activation)
    J[0,h] = ...... # Determine cost function
 Determine derivatives using the code from past Logistic Regression assignment
 Use update formula to update W2 and B2
 Loop
print(W2 and B2)

This is a simple assignment if you have code from the assignment of the last assignment. Your results (W2 and B2) should match with the result that you obtained in the Logistic Regression assignment that you did earlier. Your implementation will use NumPy and are the calculations will be using the matrix operations.  

System Parameters and Activation Functions: The W0and W1 and will be unity matrices. B0and B1 will be zero vectors. W2 and B2 will consists of ones as their elements. f0 and f1 will be linear activation functions while f2 will be the sigmoid activation function. 
