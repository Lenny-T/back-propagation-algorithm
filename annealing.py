import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# DEFINE THE NEURAL NETWORK ARCHITECTURE 
numberOfInputs = 6 # NUMBER OF INPUTS
numberOfHiddenNodes = 9 # NUMBER OF HIDDEN NODES
numberOfOutputs = 1 # NUMBER OF OUTPUTS
learningParam = 0.4 # ASSIGN THE LEARNING PARAMETER
epochs = 2000 # ASSIGN THE NUMBER OF EPOCHS
epochs += 1

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
listOfLearningParameters = []
startLearningParameter = learningParam
endLearningParameter = 0.05
# listOfLearningParameters.append(learningParam)

# T R A I N I N G
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
        originalInputWeights = hiddenInputWeights
        momentum = 0.9
        for x in range(len(hiddenInputWeights)):
            for y in range(len(hiddenInputWeights[x])):
                # hiddenInputWeights[x][y] += learningParam * (deltas[y+1][0] * inputNodes[i][x]) # ORIGINAL WEIGHT UPDATE
                if epoch == 1:
                    hiddenInputWeights[x][y] += learningParam * (deltas[y+1][0] * inputNodes[i][x]) # ORIGINAL WEIGHT UPDATE
                else:
                    hiddenInputWeights[x][y] += learningParam * (deltas[y+1][0] * inputNodes[i][x]) + (momentum * (hiddenInputWeights[x][y] - originalInputWeights[x][y])) # MOMENTUM
        # print(hiddenInputWeights)


        # UPDATE THE WEIGTHS FROM THE HIDDEN NODES TO THE OUTPUT
        # print(hiddenInputWeights)
        # print(outputWeights) # hiddenInputWeights[input][hidden]
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
        
    # A N N E A L I N G   F O R M U L A R
    listOfLearningParameters.append(learningParam)
    learningParam = (endLearningParameter + (startLearningParameter - endLearningParameter)) * (1 - (1/(1 + (np.exp(10 - ((20 * epoch)/epochs))))))

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

# PLOT THE LEARNING PARAMETER OVER EPOCHS
plt.plot(range(epochs), listOfLearningParameters)
plt.xlabel('Epochs')
plt.ylabel('Learning Parameter')
plt.title('Change In Leanring Parameters')
plt.show()

# V A L I D A T I O N






# T E S T I N G
testingData = int(0.2 * len(df))
testingData += trainingData
testInputNodes = df.iloc[trainingData+1:testingData, 1:(numberOfInputs + 1)].apply(pd.to_numeric, errors='coerce').values
testOutputNodes = df.iloc[trainingData+1:testingData, 8].apply(pd.to_numeric, errors='coerce').values

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




# PLOT ACTUAL VS PREDICTED GRAPH WITH A LINE OF BEST FIT
fig, ax = plt.subplots()
ax.scatter(predictedList, testOutputNodes)
plt.plot([min(testOutputNodes), max(testOutputNodes)], [min(testOutputNodes), max(testOutputNodes)])
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Predicted vs Actual')
plt.show()
