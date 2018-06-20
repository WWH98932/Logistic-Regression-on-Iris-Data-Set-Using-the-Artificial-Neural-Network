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

# Apply the Artificial Neural Network to Logistic Regression for Pattern Data Set 
You will use the assignment Reduce the Artificial Neural Network to Logistic Regression for Iris Train and Test as a base to do this assignment.

The data in file pattern_rand.csv has two features (X1 and X2) unlike Iris data set that had four features (petal length, petal width, sepal length, sepal width). Like Iris data set, this data set has one output with two classes (0 and 1). There are 400 examples in the file - you will use first 320 examples as training set (or m=160). The remaining 80 examples will be used as test set.

Compared to Iris data set, this data may not effectively be modeled by Logistic Regression due to nonlinearities. You will use your network that you developed in the above mentioned assignment to perform Logistic Regression on this data. As this is a single layer network, you simply will modify input layer to match with number of inputs (2 inputs 2 neurons as opposed to 4 inputs 4 neurons). Your weight matrix for Layer 1 (W1) will not be identity an matrix (because you have 2 inputs and 4 outputs so W1 will be 4x2). However, you will need to create your weight matrix in such a way that it reproduces two of four outputs exactly as inputs and the other two as zeros.

The rest of the structure will remain the same. Your program will print model parameters and accuracy for the test set.

Your implementation will use NumPy and are the calculations will be using the matrix operations as in the above mentioned assignment.
