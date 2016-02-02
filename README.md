
# NeuralNetworks

The purpose of this code is mainly educational. It is the implementation of neural network with cross entropy error function for the problem of multi-label classification.


```julia
include("NeuralNetwork.jl");
trainingData, trainingLabels, testData, testLabels = loadMnistData();
```

to train the neural network with one hidden layer of tanh followed by fully connected linear layer with softmax at the end try:
```julia
# hidden layer composed of tanh neurons
architecture = buildNetworkArchitecture(784, [50,10], [tanhComputingLayer, linearComputingLayer])
crossEntropies = Float64[]
batchSize = 20
for i = 1:10000
   minibatch = collect((batchSize*i):(batchSize*i +batchSize)) % size(trainingLabels,2) + 1 # take next 20 elements
   learningUnit = BackPropagationBatchLearningUnit(architecture, trainingData[:,minibatch ], trainingLabels[:,minibatch]);   
   updateParameters!(learningUnit, 0.1)
   if i % 100 == 0  # this one costs so lets store entropies every 100 iterations
     push!(crossEntropies, crossEntropyError(architecture, trainingData, trainingLabels))   
   end                 
end   
inferedOutputs = infer(architecture, testData)
# test accuracy
mean(mapslices(x -> indmax(x), inferedOutputs ,1)[:]  .==  mapslices(x -> indmax(x), full(testLabels),1)[:])
```
to train the neural network with one hidden layer of ReLU followed by fully connected linear layer with softmax at the end try:
```julia
# hidden layer composed of tanh neurons
architecture = buildNetworkArchitecture(784, [50,10], [reluComputingLayer, linearComputingLayer])
crossEntropies = Float64[]
batchSize = 20
for i = 1:10000
   minibatch = collect((batchSize*i):(batchSize*i +batchSize)) % size(trainingLabels,2) + 1 # take next 20 elements
   learningUnit = BackPropagationBatchLearningUnit(architecture, trainingData[:,minibatch ], trainingLabels[:,minibatch]);   
   updateParameters!(learningUnit, 0.1)
   if i % 100 == 0  # this one costs so lets store entropies every 100 iterations
     push!(crossEntropies, crossEntropyError(architecture, trainingData, trainingLabels))   
   end                 
end   
inferedOutputs = infer(architecture, testData)
# test accuracy
mean(mapslices(x -> indmax(x), inferedOutputs ,1)[:]  .==  mapslices(x -> indmax(x), full(testLabels),1)[:])
```
