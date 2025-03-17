import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# DEFINE THE NEURAL NETWORK ARCHITECTURE 
numberOfInputs = 3 # NUMBER OF INPUTS
numberOfHiddenNodes = 5 # NUMBER OF HIDDEN NODES
numberOfOutputs = 1 # NUMBER OF OUTPUTS
learningParam = 0.5 # ASSIGN THE LEARNING PARAMETER
epochs = 1000 # ASSIGN THE NUMBER OF EPOCHS

# GET THE DATA FROM THE EXCEL SHEET
df = pd.read_excel("Ouse93-96 - Student.xlsx")
trainingData = int(0.6 * len(df))
inputNodes = df.iloc[1:trainingData, 1:(numberOfInputs + 1)].apply(pd.to_numeric, errors='coerce').values # STORE THE INPUT BASED ON THE NUMBER OF INPUTS
outputNodes = df.iloc[1:trainingData, 8].apply(pd.to_numeric, errors='coerce').values # STORE THE OUTPUT BASED ION THE NUMBER OF OUTPUTS

# DATA PREPROCERSSING
inputNodes = (inputNodes - np.mean(inputNodes, axis=0)) / np.std(inputNodes, axis=0) # STANDARDISE THE INPUTS
outputNodes = (outputNodes - np.mean(outputNodes, axis=0)) / np.std(outputNodes, axis=0) # STANDARDISE THE OUTPUTS
outputNodes = (outputNodes - np.min(outputNodes)) / (np.max(outputNodes) - np.min(outputNodes))


# LECTURE EXAMPLE DATA
# inputNodes = [
#     [0.3, 0.7],
#     [0.7, 0.3],
#     [0.7, 0.2],
#     [0.6, 0.3],
#     [0.8, 0.3],
#     [0.7, 0.4],
#     [0.3, 0.6],
#     [0.4, 0.7],
#     [0.2, 0.7],
#     [0.3, 0.8],
#     [0, 1],
#     [1, 0],
#     [0.5, 0.5],
#     [0.6, 0.2],
#     [0.8, 0.4],
#     [0.2, 0.6],
#     [0.4, 0.8]
# ]
# outputNodes = [0.102, 0.074, 1.024, 0.975, 0.975, 1.025, 0.975, 1.026, 1.001, 0.960, 1.011, 0.998, 1.123, 1.125, 1.124, 1.125, 1.124]


def sigmoidFunction(weightedSum):
    return 1 / (1 + np.exp(-weightedSum)) # Uj = 1 / 1 + e^Sj
    # return ((np.exp(weightedSum) - np.exp(-weightedSum)) / (np.exp(weightedSum) + np.exp(-weightedSum))) # Tan(h)

# CREATE THE SIGMOID DERIVATIVE FUNCTION
def sigmoidDerivative(nodeValue):
    return nodeValue * (1 - nodeValue) # Uj (1 - Uj)
    # return (1 - (nodeValue**2)) # Derivative of # Tan(h)

# INITIALSIED THE WEIGHTS AND BIASES RANDOMLY
hiddenInputWeights = np.zeros((numberOfInputs, numberOfHiddenNodes)) # INITIALISED WEIGHTS MATRIX TO 0 np.zeros((input_size, hidden_nodes))
hiddenBiases = np.zeros((numberOfHiddenNodes)) # INITIALISE BIAS MATRIX TO 0
# LOOP THROUGH THE NUMBER OF INPUT NODES
for i in range(numberOfInputs):
    # LOOP THROUGH THE NUMBER OF HIDDEN NODES
    for j in range(numberOfHiddenNodes):
        hiddenInputWeights[i,j] = np.random.randn() # ASSIGN WEIGHT[I, J] WITH A RANDOM WEIGHT np.random.randn()
hiddenInputWeights = np.round(hiddenInputWeights, 4)

# LOOP THROUGH THE NUMBER OF HIDDEN NODES
for i in range(numberOfHiddenNodes):
    hiddenBiases[i] = np.random.randn() # ASSIGN BIAS[J] WITH A RANDOM WEIGHT np.random.randn()
hiddenBiases = np.round(hiddenBiases, 4)
# print(f"Weights From Inputs To Hidden:\n{hiddenInputWeights}")
# print(f"Hidden Biases:\n{hiddenBiases}")

# INITIALISE THE OUTPUT WEIGHTS AND BIAS RANDOMLY
outputWeights = np.random.randn(numberOfHiddenNodes, numberOfOutputs) # GET THE RANDOM WEIGHTS OF THE OUTPUT NODE
outputWeights = np.round(outputWeights, 4)

outputBias = np.random.randn() # GET THE RANDOM BIAS OF THE OUTPUT NODE
outputBias = np.round(outputBias, 4)
# print(f"Weights From Hidden To Output:\n{outputWeights}")
# print(f"Output Biases:\n{outputBias}")

# LOOP THROUGH THE NUMBER OF EPOCHS
listOfErrors = []
listOfMSE = []
for epoch in range(epochs):
    # LOOP THROUGH ALL OF THE INPUTS
    totalError = 0 # RESET TOTAL ERROR
    for i in range(len(inputNodes)):
        # F O R W A R D   P A S S 
        weightedSum = np.dot(inputNodes[i], hiddenInputWeights) + hiddenBiases # CALCULATE THE WIEGHTED SUM USING weighted_sum = np.dot(inputs, weights) + biases.
        weightedSum = np.round(weightedSum, 4)
        Uj = sigmoidFunction(weightedSum) # CALCULATE THE SIGMOID FUNCTION USING THE WIEGHTED SUM. THESE ARE THE NEW NODE VALUES FOR THE HIDDEN NODES.
        predictedOutputWeightesSum = np.dot(Uj, outputWeights) + outputBias # CALCULATE THE PREDICTED OUTPUT USING np.dot(hidden_activations, output_weights) + output_bias
        predictedOutput = sigmoidFunction(predictedOutputWeightesSum)
        error = outputNodes[i] - predictedOutput # CALCULATE THE ERROR USING ACTUAL OUTPUT - PREDICTED OUTPUT
        # listOfErrors.append(error) # ADD ERROR TO AN ARRAY FOR THE GRAPH
        # mse = np.mean(error ** 2) # CALCULATE THE MEAN SQUARED ERROR USING np.mean(errors ** 2)
        totalError += (error ** 2) # CALCULATE THE TOTAL ERROR FOR THE EPOCH
        # print(f"{outputNodes[i]} - {outputWeights}")

        # B A C K W A R D   P A S S
        deltas = {} # DICTONARY OF OUTPUTS
        outputFirstDerivative = sigmoidDerivative(predictedOutput) # CALCULATE THE FIRST DERIVATIVE OF THE OUTPUT NODE
        deltas["Output"] = (outputNodes[i] - predictedOutput) * outputFirstDerivative # CALCULATE DELTA OF OUTPUT
        # print(f"Deltas: {deltas}")
        # LOOP THROUGH ALL THE HIDDEN NODES
        # print(f"Output Weight: {outputWeights[2][0]}")
        # print(f"Node Value: {Uj[2]}")
        for j in range(len(Uj)):
            hiddenNodeFirstDerivative = sigmoidDerivative(Uj[j]) # CALCULATE THE FIRST DERIVATIVE OF THE HIDDEN NODES
            hiddenNodeDelta = (outputWeights[j][0] * deltas["Output"]) * hiddenNodeFirstDerivative # CALCULATE THE DELTA OF THE HIDDEN NODE
            deltas[(j+1)] = hiddenNodeDelta
        

        # UPDATE THE WEIGHTS FROM THE INPUTS
        for x in range(len(hiddenInputWeights)):
            for y in range(len(hiddenInputWeights[x])):
                hiddenInputWeights[x][y] += learningParam * (deltas[y+1][0] * inputNodes[i][x])
        # print(hiddenInputWeights)

        # UPDATE THE WEIGTHS FROM THE HIDDEN NODES TO THE OUTPUT
        # print(hiddenInputWeights)
        # print(outputWeights) # hiddenInputWeights[input][hidden]
        for x in range(len(outputWeights)):
            outputWeights[x, 0] += learningParam * (deltas["Output"][0] * Uj[x])
        
        # BIAS UPDATES
        for i in range(numberOfHiddenNodes):
            hiddenBiases[i] += learningParam * deltas[i+1][0]
        outputBias += learningParam * (deltas["Output"][0] * predictedOutput[0])
        
    # APPEND MSE TO THE LIST FOR PLOTTING
    listOfMSE.append(totalError/len(inputNodes))
    if (epoch % 100 == 0):
        print(f"Epoch: {epoch}")
        print(f"MSE: {totalError/len(inputNodes)}")

# PLOT THE MSE OVER EPOCHS
plt.plot(range(epochs), listOfMSE)
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Training Error Over Epochs')
plt.show()