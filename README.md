
# NeuralNetworks

The purpose of this code is mainly educational. It is the implementation of neural network with cross entropy error function for the problem of multi-label classification.


```julia
include("NeuralNetwork.jl");
trainingData, trainingLabels, testData, testLabels = loadMnistData();
```

to train the neural network with one hidden layer of tanh followed by fully connected linear layer with softmax at the end (using Adam) try:
```julia
architecture = buildNetworkArchitecture(784, [10], [linearComputingLayer])
adam = AdamOptimizer(0, 0.002, 0.9, .999, architecture)
crossEntropiesAdam = Float64[]
batchSize = 128
for i = 1:40000
   minibatch = collect((batchSize*i):(batchSize*i +batchSize)) % size(trainingLabels,2) + 1 # take next 20 elements
   learningUnit = BackPropagationBatchLearningUnit(architecture, trainingData[:,minibatch ],
                                                   trainingLabels[:,minibatch]);
   adam.updateRule!(learningUnit, adam.params)
   if i % 100 == 0  # this one costs so lets store entropies every 100 iterations
     push!(crossEntropiesAdam, crossEntropyError(architecture, trainingData, trainingLabels))
   end
end
```
to train the neural network with one hidden layer of ReLU followed by fully connected linear layer with softmax at the end (using Momentum) try:
```julia
architecture = buildNetworkArchitecture(784, [10], [linearComputingLayer])
momentum = MomentumOptimizer(0.05, 0.9, architecture)
crossEntropiesMomentum = Float64[]
batchSize = 128
for i = 1:40000
   println(i)
   minibatch = collect((batchSize*i):(batchSize*i +batchSize)) % size(trainingLabels,2) + 1 # take next 20 elements
   learningUnit = BackPropagationBatchLearningUnit(architecture, trainingData[:,minibatch ],
                                                  trainingLabels[:,minibatch]);
   momentum.updateRule!(learningUnit, momentum.params)
   if i % 100 == 0  # this one costs so lets store entropies every 100 iterations
     push!(crossEntropiesMomentum, crossEntropyError(architecture, trainingData, trainingLabels))
   end
end
```
