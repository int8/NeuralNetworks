# NeuralNetworks

The purpose of this code is mainly educational. It is the implementation of neural network with cross entropy error function for the problem of multi-label classification.


```julia
include("NeuralNetwork.jl");

# Pkg.add("RDatasets") - install RDatasets if you haven't already
# Phg.add("Gadfly") - install gadfly for visualization purposes
using Gadfly

architecture = buildNetworkArchitectureWithOneHiddenSigmoids([784,50, 10]) # 50 neurons in a hidden layer now
crossEntropies = Float64[]
batchSize = 20
for i = 1:30000
   minibatch = collect((batchSize*i):(batchSize*i +batchSize)) % size(trainingLabels,2) + 1 # take next 20 elements
   learningUnit = BackPropagationBatchLearningUnit(architecture, trainingData[:,minibatch ],
                                                   trainingLabels[:,minibatch]);   
   updateParameters!(learningUnit, 0.3)  
   if i % 100 == 0  # this one costs so lets store entropies every 100 iterations
     push!(crossEntropies, crossEntropyError(architecture, trainingData, trainingLabels))   
   end                 
end   
plot(x = 1:length(crossEntropies), y = crossEntropies, Geom.line, Guide.xlabel("iterations"), Guide.ylabel("error"))
```
