module Minds

export Mind, InputLayer, OutputLayer, HiddenLayer,
        learn!, predict, 
        relu, σ, softmax, cross_entropy


abstract type Layer
end

mutable struct InputLayer <: Layer
    nodes::Int
end

mutable struct OutputLayer <: Layer
    nodes::Int
    weights::Matrix{Float32}
    biases::Vector{Float32}
    f::Function
    score::Function
    df::Function
    learning::Bool
    λ::Float32
end
function OutputLayer(n; f=softmax, score=cross_entropy, learning=true, λ=0.01)
    return OutputLayer(n, Matrix{Float32}(undef,0,0), Vector{Float32}(undef,0), 
                        f, score, (P,Y)->((P .- Y) / size(P,2)), learning, λ)
end

mutable struct HiddenLayer <: Layer
    nodes::Int
    weights::Matrix{Float32}
    biases::Vector{Float32}
    f::Function
    df::Function
    learning::Bool
    λ::Float32
end
function HiddenLayer(n; f=relu, learning=true, λ=0.01)
    return HiddenLayer(n, Matrix{Float32}(undef,0,0), Vector{Float32}(undef,0), 
                        f, Z->d(f)(Z), learning, λ)
end

mutable struct ConvolutionalLayer <: Layer
    filterx::Int
    filtery::Int 
    depth::Int
    imagex::Int
    imagey::Int
    n::Int
    weights::Matrix{Float32}
    biases::Vector{Float32}
    f::Function
    df::Function
    learning::Bool
    λ::Float32
end
function ConvolutionalLayer(filterx, filtery, depth, imagex, imagey, ; f=relu, learning=true, λ=0.01)
    return ConvolutionalLayer(n, filterx, filtery, depth, imagex, imagey, Matrix{Float32}(undef,0,0), Vector{Float32}(undef,0), 
                        f, Z->d(f)(Z), learning, λ)
end

mutable struct Mind
    layers::Vector{Layer}
end

function Mind(layers)
    for (l_out, l_in) in zip(layers[2:end], layers[1:end-1])
        if typeof(l_out) == HiddenLayer
            l_out.weights = randn(l_out.n, l_in.n) / √l_in.n
            l_out.biases = randn(l_out.n)
        elseif typeof(l_out) == ConvolutionalLayer
            l_out.weights = randn(filterx*filtey*depth, l_in.n) / √l_in.n
            l_out.biases = randn(nout)
        elseif typeof(l_out) == OutputLayer
            l_out.weights = randn(l_out.n, l_in.n) / √l_in.n
            l_out.biases = randn(l_out.n)
            return Mind(layers)
        end
    end
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
        backprop!(mind, x, y, 1)
        push!(training_skorz, mind.layers[end].score(predict(mind, X), Y))
        push!(test_skorz, mind.layers[end].score(predict(mind, X2), Y2))
    end
    return training_skorz, test_skorz
end

function learn!(mind::Mind, X::Matrix{Float32}, Y::Matrix{Float32}, cycles::Int)
    training_skorz = []
    for cycle ∈ 1:cycles
        backprop!(mind, X,Y, 1)
        push!(training_skorz, mind.layers[end].score(predict(mind, X), Y))
    end
    return training_skorz
end

end # module
