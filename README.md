# NeuralNetworks

The purpose of this code is mainly educational. It is the implementation of neural network with cross entropy error function for the problem of multi-label classification.

Quick example of how to use the code:

```julia
# add mnist package if not added yet
# Pkg.add("MNIST")
```

Include the code and load the datasets.

```julia
include("NeuralNetwork.jl");
# load MNIST datasets
trainingData, trainingLabels, testData, testLabels = loadMnistData();
```

Now, to train the neural network with one hidden layer of tanh followed by fully connected linear layer with softmax at the end (using Adam) try:
```julia
architecture = buildNetworkArchitecture(784, [50, 10], [tanhComputingLayer, linearComputingLayer])
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
Or, to train the neural network with one hidden layer of sigmoid followed by fully connected linear layer with softmax at the end (using Momentum) try:
```julia
architecture = buildNetworkArchitecture(784, [50,10], [sigmoidComputingLayer, linearComputingLayer])
momentum = MomentumOptimizer(0.05, 0.9, architecture)
crossEntropiesMomentum = Float64[]
batchSize = 128
for i = 1:40000   
   minibatch = collect((batchSize*i):(batchSize*i +batchSize)) % size(trainingLabels,2) + 1 # take next 20 elements
   learningUnit = BackPropagationBatchLearningUnit(architecture, trainingData[:,minibatch ],
                                                  trainingLabels[:,minibatch]);
   momentum.updateRule!(learningUnit, momentum.params)
   if i % 100 == 0  # this one costs so lets store entropies every 100 iterations
     push!(crossEntropiesMomentum, crossEntropyError(architecture, trainingData, trainingLabels))
   end
end
```

Alternative optimization techniques are: ```AdaGradOptimizer, AdaDeltaOptimizer, SGDOptimizer ```. The types of layer that has been implemented include:
``` reluComputingLayer, tanhComputingLayer, sigmoidComputingLayer, linearComputingLayer ```

Please visit http://int8.io/comparison-of-optimization-techniques-stochastic-gradient-descent-momentum-adagrad-and-adadelta for details (+ to see some experiments)
