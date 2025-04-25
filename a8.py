from neural import *

print("\n\nTraining XOR\n\n")
xor_trainingData = [
    ([0,0], [0]),
    ([0,1], [1]),
    ([1,0], [1]),
    ([1,1], [0]),
]
partyAffiliation_trainingData = [
    ([.9,.6,.8,.3,.1], [1]),
    ([.8,.8,.4,.6,.4], [1]),
    ([.7,.2,.4,.6,.3], [1]),
    ([.5,.5,.8,.4,.8], [0]),
    ([.3,.1,.6,.8,.8], [0]),
    ([.6,.3,.4,.3,.6], [0])
]
inputs = [
    ([1,1,1,.1,.1]),
    ([.5,.2,.1,.7,.8]),
    ([.8,.3,.3,.3,.8]),
    ([.8,.3,.3,.8,.3]),
    ([.9,.8,.8,.3,.6])
]
# neuralNetwork = NeuralNet(2,2,1)
# neuralNetwork = NeuralNet(2,8,1)
# neuralNetwork = NeuralNet(2,1,1)
# neuralNetwork.train(xor_trainingData)
neuralNetwork = NeuralNet(5,16,1)
neuralNetwork.train(partyAffiliation_trainingData) 
print(neuralNetwork.test(inputs))
print("<<<<<<<<<<<<<< XOR >>>>>>>>>>>>>>\n")




