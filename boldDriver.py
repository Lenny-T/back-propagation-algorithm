import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# DEFINE THE NEURAL NETWORK ARCHITECTURE 
numberOfInputs = 6 # NUMBER OF INPUTS
numberOfHiddenNodes = 9 # NUMBER OF HIDDEN NODES
numberOfOutputs = 1 # NUMBER OF OUTPUTS
learningParam = 0.4 # ASSIGN THE LEARNING PARAMETER
epochs = 100 # ASSIGN THE NUMBER OF EPOCHS
epochs += 1

# INITIALISATION FOR BOLD DRIVER
initialLearningParam = 0.4
learningParam = initialLearningParam # ASSIGN THE LEARNING PARAMETER
leanringParamIncrease = 1.05  # 5% increase when error decreases
leanringParamDecrease = 0.3   # 50% decrease when error increases
previousError = float('inf')  # Track error from previous epoch

# GET THE DATA FROM THE EXCEL SHEET
df = pd.read_excel("Important-Files/Ouse93-96 - Student (Cleaned).xlsx")
trainingData = int(0.6 * len(df))
inputNodes = df.iloc[1:trainingData, 1:(numberOfInputs + 1)].apply(pd.to_numeric, errors='coerce').values # STORE THE INPUT BASED ON THE NUMBER OF INPUTS
outputNodes = df.iloc[1:trainingData, 8].apply(pd.to_numeric, errors='coerce').values # STORE THE OUTPUT BASED ION THE NUMBER OF OUTPUTS

# DATA PREPROCERSSING
inputMean = np.mean(inputNodes, axis=0)
inputStandardDeviation = np.std(inputNodes, axis=0)
inputNodes = (inputNodes - inputMean) / inputStandardDeviation # STANDARDISE THE INPUTS

outputMean = np.mean(outputNodes, axis=0)
outputStandardDeviation = np.std(outputNodes, axis=0)
outputNodes = (outputNodes - outputMean) / outputStandardDeviation # STANDARDISE THE OUTPUTS

outputMin = np.min(outputNodes, axis=0)
outputMax = np.max(outputNodes, axis=0)
outputNodes = (outputNodes - outputMin) / (outputMax - outputMin) # NORMALISE OUTPUT DATA


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


# T R A I N I N G
for epoch in range(epochs):
    # LOOP THROUGH ALL OF THE INPUTS
    totalError = 0 # RESET TOTAL ERROR
    for i in range(len(inputNodes)):
        index = 1
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



        # M O M E N T U M
        # UPDATE THE WEIGHTS FROM THE INPUTS
        originalInputWeights = hiddenInputWeights
        momentum = 0.9
        for x in range(len(hiddenInputWeights)):
            for y in range(len(hiddenInputWeights[x])):
                if epoch == 1:
                    hiddenInputWeights[x][y] += learningParam * (deltas[y+1][0] * inputNodes[i][x]) # ORIGINAL WEIGHT UPDATE
                else:
                    hiddenInputWeights[x][y] += learningParam * (deltas[y+1][0] * inputNodes[i][x]) + (momentum * (hiddenInputWeights[x][y] - originalInputWeights[x][y])) # MOMENTUM


        # UPDATE THE WEIGTHS FROM THE HIDDEN NODES TO THE OUTPUT
        originalOutputWeights = outputWeights
        for x in range(len(outputWeights)):
            if epoch == 1:
                outputWeights[x, 0] += learningParam * (deltas["Output"][0] * Uj[x]) # ORIGINAL WEIGHT UPDATE
            else:
                outputWeights[x, 0] += learningParam * (deltas["Output"][0] * Uj[x]) + (momentum * (outputWeights[x][0] - originalOutputWeights[x][0])) # MOMENTUM
        
        # BIAS UPDATES
        originialHiddenBiases = hiddenBiases
        for i in range(numberOfHiddenNodes):
            if epoch == 1:
                hiddenBiases[i] += learningParam * deltas[i+1][0] # ORIGINAL WEIGHT UPDATE
            else:
                hiddenBiases[i] += learningParam * deltas[i+1][0] + (momentum * (hiddenBiases[i] - originialHiddenBiases[i])) # MOMENTUM
        
        originalOutputBias = outputBias
        if epoch == 1:
            outputBias += learningParam * (deltas["Output"][0] * predictedOutput[0]) # ORIGINAL UPDATING OF WEIGHTS
        else:
            outputBias += learningParam * (deltas["Output"][0] * predictedOutput[0]) + (momentum * (outputBias - originalOutputBias)) # MOMENTUM


        # B O L D   D R I V E R
        # STORING THE CURRENT AND PREVIOUS MSE
        # IF MSE INCREASED MORE THAN 4% THEN REDUCE THE LEARNING PARAMETER BY 30%
        # CHECK AGAIN IF LEARNING PARAMETER HAS INCREASED / DECREASED
        # IF MSE DECREASED MORE THAN 4% THEN INCREASE THE LEARNING PARAMETER BY 5%.
        # CHECK AGAIN IF LEARNING PARAMETER HAS INCREASED / DECREASED


    # APPEND MSE TO THE LIST FOR PLOTTING
    listOfMSE.append(totalError/len(inputNodes))
    if (epoch % 100 == 0):
        print(f"Epoch: {epoch}")
        print(f"MSE: {totalError/len(inputNodes)}")

# V A L I D A T I O N
validationData = int(0.2 * len(df))
validationData += trainingData
validationInputNodes = df.iloc[trainingData+1:validationData, 1:(numberOfInputs + 1)].apply(pd.to_numeric, errors='coerce').values
validationOutputNodes = df.iloc[trainingData+1:validationData, 8].apply(pd.to_numeric, errors='coerce').values

# DATA PREPROCESSING
validationInputNodes = (validationInputNodes - inputMean) / inputStandardDeviation
validationOutputNodes = (validationOutputNodes - outputMean) / outputStandardDeviation
validationOutputNodes = (validationOutputNodes - outputMin) / (outputMax - outputMin)
totalError = 0
predictedList = []

for i in range(len(validationInputNodes)):
    weightedSum = np.dot(validationInputNodes[i], hiddenInputWeights) + hiddenBiases # CALCULATE THE WIEGHTED SUM USING weighted_sum = np.dot(inputs, weights) + biases.
    weightedSum = np.round(weightedSum, 4)
    Uj = sigmoidFunction(weightedSum) # CALCULATE THE SIGMOID FUNCTION USING THE WEIGHTED SUM. THESE ARE THE NEW NODE VALUES FOR THE HIDDEN NODES.
    predictedOutputWeightesSum = np.dot(Uj, outputWeights) + outputBias # CALCULATE THE PREDICTED OUTPUT USING np.dot(hidden_activations, output_weights) + output_bias
    predictedOutput = sigmoidFunction(predictedOutputWeightesSum)
    predictedList.append(predictedOutput)
    error = validationOutputNodes[i] - predictedOutput # CALCULATE THE ERROR USING ACTUAL OUTPUT - PREDICTED OUTPUT
    # print(f"Predicted: {predictedOutput}, Correct: {validationOutputNodes[i]}")
    totalError += (error ** 2)
print(f"Validation MSE: {totalError/len(validationInputNodes)}")


# T E S T I N G
testingData = int(0.2 * len(df))
testingData += validationData
testInputNodes = df.iloc[validationData+1:testingData, 1:(numberOfInputs + 1)].apply(pd.to_numeric, errors='coerce').values
testOutputNodes = df.iloc[validationData+1:testingData, 8].apply(pd.to_numeric, errors='coerce').values

# DATA PREPROCESSING
testInputNodes = (testInputNodes - inputMean) / inputStandardDeviation
testOutputNodes = (testOutputNodes - outputMean) / outputStandardDeviation
testOutputNodes = (testOutputNodes - outputMin) / (outputMax - outputMin)
totalError = 0
predictedList = []

for i in range(len(testInputNodes)):
    weightedSum = np.dot(testInputNodes[i], hiddenInputWeights) + hiddenBiases # CALCULATE THE WIEGHTED SUM USING weighted_sum = np.dot(inputs, weights) + biases.
    weightedSum = np.round(weightedSum, 4)
    Uj = sigmoidFunction(weightedSum) # CALCULATE THE SIGMOID FUNCTION USING THE WEIGHTED SUM. THESE ARE THE NEW NODE VALUES FOR THE HIDDEN NODES.
    predictedOutputWeightesSum = np.dot(Uj, outputWeights) + outputBias # CALCULATE THE PREDICTED OUTPUT USING np.dot(hidden_activations, output_weights) + output_bias
    predictedOutput = sigmoidFunction(predictedOutputWeightesSum)
    predictedList.append(predictedOutput)
    error = testOutputNodes[i] - predictedOutput # CALCULATE THE ERROR USING ACTUAL OUTPUT - PREDICTED OUTPUT
    # print(f"Predicted: {predictedOutput}, Correct: {testOutputNodes[i]}")
    totalError += (error ** 2)
print(f"TEST MSE: {totalError/len(testInputNodes)}")


# P L O T   A L L   T H E   G R A P H S
# PLOT THE MSE OVER EPOCHS
plt.plot(range(epochs), listOfMSE)
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Training Error Over Epochs')
plt.show()

# PLOT ACTUAL VS PREDICTED GRAPH WITH A LINE OF BEST FIT
fig, ax = plt.subplots()
ax.scatter(predictedList, testOutputNodes)
plt.plot([min(testOutputNodes), max(testOutputNodes)], [min(testOutputNodes), max(testOutputNodes)])
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Predicted vs Actual')
plt.show()