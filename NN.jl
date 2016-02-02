using MNIST
abstract Layer

function loadMnistData()

  a,b = MNIST.traindata()
  t,l = MNIST.testdata()
  t = (t .- mean(t,2)) / std(t .- mean(t,2))
  a = (a .- mean(a,2)) / std(a .- mean(a,2))
  b = sparse(convert(Array{Int64}, b + 1),1:60000, [ 1 for i in 1:60000])
  l = sparse(convert(Array{Int64}, l + 1),1:10000, [ 1 for i in 1:10000])
  return(a,b,t,l)
end

function appendColumnOfOnes(a::Array{Float64,2})
  vcat(a,ones(1,size(a ,2)))
end

function sigmoidNeuronTransformFunction(params, input)
  return 1.0 ./ (1.0 .+ exp(-params * appendColumnOfOnes(input)))
end


function linearNeuronTransformFunction(params, input)
  return params * appendColumnOfOnes(input)
end



function exponentialNormalizer(params, input)
  denominator = sum(exp(input),1)
  return exp(input) ./ denominator
end



type FullyConnectedComputingLayer <: Layer
  inputSize::Int64
  numberOfNeurons::Int64
  parameters::Array{Float64,2}
  transform::Function
  derivative::Function # derivative added here

  function FullyConnectedComputingLayer(inputSize, numberOfNeurons, transform::Function, derivative::Function)
    parameters = randn(numberOfNeurons, inputSize + 1)  * 0.1 # adding one param column for bias
    return new(inputSize, numberOfNeurons, parameters, transform, derivative)
  end
end


type SoftMaxLayer <: Layer
  numberOfNeurons::Int64
  parameters::Any
  transform::Function

  function SoftMaxLayer(numberOfNeurons)
    return new(numberOfNeurons, [], exponentialNormalizer)
  end
end

type NetworkArchitecture
  layers::Array{Layer}
  function NetworkArchitecture(firstLayer::Layer)
    return new([firstLayer])
  end
end


function addSoftMaxLayer(architecture::NetworkArchitecture)
 lastNetworkLayer = architecture.layers[end]
 numberOfNeurons = lastNetworkLayer.numberOfNeurons
 softMaxLayer = SoftMaxLayer(numberOfNeurons)
 push!(architecture.layers, softMaxLayer)
end


function addFullyConnectedSigmoidLayer(arch::NetworkArchitecture, numberOfNeurons::Int64)
 lastNetworkLayer = arch.layers[end]
 inputSize = lastNetworkLayer.numberOfNeurons
 #derivative added
 sigmoidLayer = FullyConnectedComputingLayer(inputSize, numberOfNeurons, sigmoidNeuronTransformFunction, x -> x .* (1 - x))
 push!(arch.layers, sigmoidLayer)
end


function addFullyConnectedLinearLayer(architecture::NetworkArchitecture, numberOfNeurons::Int64)
 lastNetworkLayer = architecture.layers[end]
 inputSize = lastNetworkLayer.numberOfNeurons
 linearLayer = FullyConnectedComputingLayer(inputSize, numberOfNeurons, linearNeuronTransformFunction, x -> 1)
 push!(architecture.layers, linearLayer)
end


function infer(architecture::NetworkArchitecture, input)
  currentResult = input
  for i in 1:length(architecture.layers)
     layer = architecture.layers[i]
     currentResult = layer.transform(layer.parameters, currentResult)
  end
  return currentResult
end

function crossEntropyError(architecture::NetworkArchitecture, input, labels)
 probabilitiesSparseMatrix = infer(architecture, input) .* labels
 probabilities = sum(probabilitiesSparseMatrix , 1)
 return -mean(log(probabilities))
end


type BackPropagationBatchLearningUnit
  networkArchitecture::NetworkArchitecture
  dataBatch::Array{Float64,2}
  labels::AbstractSparseMatrix
  outputs::Array{Array{Float64,2}} # outputs remembered now
  deltas::Array{Array{Float64,2}} # deltas kept here

  function BackPropagationBatchLearningUnit(arch::NetworkArchitecture, dataBatch::Array{Float64,2}, labels::AbstractSparseMatrix)
     outputs = [ zeros(l.numberOfNeurons, size(dataBatch,2)) for l in arch.layers ]
     deltas = [ zeros(l.numberOfNeurons, size(dataBatch,2)) for l in arch.layers ]
     return new(arch, dataBatch, labels, outputs, deltas)
  end
end


function forwardPass!(learningUnit::BackPropagationBatchLearningUnit)
  currentResult = learningUnit.dataBatch
  for i in 1:length(learningUnit.networkArchitecture.layers)
     layer = learningUnit.networkArchitecture.layers[i]
     currentResult = layer.transform(layer.parameters, currentResult)
     learningUnit.outputs[i]  = currentResult
  end
end


function backwardPass!(learningUnit::BackPropagationBatchLearningUnit)

  layer = learningUnit.networkArchitecture.layers[end-1]
  learningUnit.deltas[end-1]  = layer.derivative(learningUnit.outputs[end-1]) .*  (learningUnit.outputs[end] - learningUnit.labels)

  for i in 2:(length(learningUnit.networkArchitecture.layers) - 1)
      higherLayer = learningUnit.networkArchitecture.layers[end - i + 1]
      currentLayer = learningUnit.networkArchitecture.layers[end - i]
      learningUnit.deltas[end-i] = currentLayer.derivative(learningUnit.outputs[end-i]) .* (transpose(higherLayer.parameters[:,(1:end-1)]) * learningUnit.deltas[end - i + 1])
  end
end

function updateParameters!(unit::BackPropagationBatchLearningUnit, learningRate)
  forwardPass!(unit)
  backwardPass!(unit)
  derivativeW= (unit.deltas[1] * transpose(unit.dataBatch)) / size(unit.dataBatch,2);
  unit.networkArchitecture.layers[1].parameters[:,1:(end-1)] = unit.networkArchitecture.layers[1].parameters[:,1:(end-1)] - learningRate * derivativeW;
  derivativeB = mean(unit.deltas[1],2);
  unit.networkArchitecture.layers[1].parameters[:,end] =  unit.networkArchitecture.layers[1].parameters[:,end] - learningRate * derivativeB;
  for i in 2:(length(unit.networkArchitecture.layers) - 1)
    derivativeW = (unit.deltas[i] * transpose(unit.outputs[i-1])) / size(unit.dataBatch,2);
    unit.networkArchitecture.layers[i].parameters[:,1:(end-1)] = unit.networkArchitecture.layers[i].parameters[:,1:(end-1)] - learningRate * derivativeW;
    derivativeB = mean(unit.deltas[i],2);
    unit.networkArchitecture.layers[i].parameters[:,end] =  unit.networkArchitecture.layers[i].parameters[:,end] - learningRate * derivativeB;
  end
end

# helper to build SoftMax architecture
function buildNetworkArchitectureSoftMax(sizes)
  firstLayer = FullyConnectedComputingLayer(sizes[1], sizes[2], linearNeuronTransformFunction, x -> 1);
  architecture = NetworkArchitecture(firstLayer);
  addSoftMaxLayer(architecture)
  return(architecture)
end

# helper to build an architecture with hidden sigmoid layers
function buildNetworkArchitectureWithOneHiddenSigmoids(sizes)
  firstLayer = FullyConnectedComputingLayer(sizes[1], sizes[2], sigmoidNeuronTransformFunction, x -> x .* (1 - x));
  architecture = NetworkArchitecture(firstLayer);
  for i in 3:(length(sizes)-1)
    addFullyConnectedSigmoidLayer(architecture, sizes[i]);
  end
  addFullyConnectedLinearLayer(architecture, sizes[end]);
  addSoftMaxLayer(architecture)
  return(architecture)
end
