module Minds

export Mind, ImputLayer, OutputLayer, HiddenLayer,
        learn!, predict, 
        relu, σ, softmax, cross_entropy


abstract type Layer
end

mutable struct InputLayer <: Layer
    nodes::Int
end

mutable struct OutputLayer <: Layer
    nodes::Int
    f::Function
    score::Function
    df::Function
    learning::Bool
    λ::Float32
end
OutputLayer(n, f, score, learning=true, λ=0.01) = OutputLayer(n, f, score, (P,Y)->((P .- Y) / size(P,2)), learning, λ)


mutable struct HiddenLayer <: Layer
    nodes::Int
    f::Function
    df::Function
    learning::Bool
    λ::Float32
end
HiddenLayer(n, f, learning=true, λ=0.01) = HiddenLayer(n, f, Z->d(f)(Z), learning, λ)

mutable struct Mind
    layers::Vector{Layer}
    weights::Vector{Matrix{Float32}}
    biases::Vector{Vector{Float32}}
end

function Mind(layers)
    l =  length(layers) - 1
    ws = Vector{Matrix{Float32}}(undef,l)
    bs = Vector{Vector{Float32}}(undef, l)
    for i ∈ 1:l
        nout = layers[i+1].nodes 
        nin = layers[i].nodes 
        ws[i] = randn(nout, nin) / √nin
        bs[i] = randn(nout)
    end
    return Mind(layers, ws, bs)
end

relu(X) = max.(X, 0)

σ(X) = 1 ./ (1 .+ exp.(-X))

function d(f::Function)
    if f == relu
        drelu(Z) = sign.(Z)
        return drelu
    elseif f == σ
        dσ(Z) = Z .* (1 .- Z)
        return dσ
    end
end

function softmax(X)
    E = exp.(X)
    M = sum(E, dims=1)
    return E ./ M
end

function backprop!(mind::Mind, X::Matrix{Float32}, Y::Matrix{Float32}, l=1)
    if typeof(mind.layers[l]) == InputLayer 
        return backprop!(mind, X, Y, l+1)
    elseif typeof(mind.layers[l]) == OutputLayer 
        Z = mind.layers[l].f(mind.weights[l-1]*X .+ mind.biases[l-1])
        δ = mind.layers[l].df(Z, Y)
        dZ = 1
    else
        Z = mind.layers[l].f(mind.weights[l-1]*X .+ mind.biases[l-1])
        δ = backprop!(mind, Z, Y, l+1)
        dZ = mind.layers[l].df(Z)
    end
    ∂C = mind.weights[l-1]' * (δ .* dZ)
    if mind.layers[l].learning
        mind.biases[l-1] .-= mind.layers[l].λ * sum(δ, dims=2)
        mind.weights[l-1] .-=  mind.layers[l].λ * (δ * X')
    end
    return ∂C
end

function predict(mind::Mind, X::Matrix{Float32}, l=1)
    if typeof(mind.layers[l]) == InputLayer 
        return predict(mind, X, l+1)
    elseif typeof(mind.layers[l]) == OutputLayer 
        return mind.layers[l].f(mind.weights[l-1]*X .+ mind.biases[l-1])
    else
        Z = mind.layers[l].f(mind.weights[l-1]*X .+ mind.biases[l-1])
        return predict(mind, Z, l+1)
    end
end

cross_entropy(P::Matrix{Float32}, Y::Matrix{Float32}) = -sum(Y .* log.(P .+ eps())) / size(Y,2)

function batch(N, n)
    random_sort = sort!([(rand(),i) for i ∈ 1:N])
    randomized_indices = [r[2] for r ∈ random_sort]
    chunks = Int(round(N//n))
    segmentation = N//chunks
    begendices = [Int(floor(segmentation*i)) for i ∈ 0:chunks]
    return [randomized_indices[a+1:b] for 
            (a,b) ∈ zip(begendices[1:end-1].+1, begendices[2:end])]
end

function learn!(mind::Mind, X::Matrix{Float32}, Y::Matrix{Float32},  
                X2::Matrix{Float32}, Y2::Matrix{Float32}, cycles::Int)
    training_skorz = []
    test_skorz = []
    for cycle ∈ 1:cycles
        for randindices = batch(size(X,2), 128)
            x = X[:, randindices]
            y = Y[:, randindices]
            backprop!(mind, x, y, 1)
        end
        push!(training_skorz, mind.layers[end].score(predict(mind, X), Y))
        push!(test_skorz, mind.layers[end].score(predict(mind, X2), Y2))
    end
    return training_skorz, test_skorz
end

end # module
