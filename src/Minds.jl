module Minds

export Mind, InputLayer, OutputLayer, HiddenLayer, interconnect,
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
    nodes::Int
    weights::Matrix{Float32}
    biases::Vector{Float32}
    f::Function
    df::Function
    learning::Bool
    λ::Float32
end
function ConvolutionalLayer(filterx, filtery, depth, imagex, imagey; f=relu, learning=true, λ=0.01)
    return ConvolutionalLayer(filterx, filtery, depth, imagex, imagey, (imagex-filterx+1)*(imagey-filtery+1)*depth,
                                Matrix{Float32}(undef,0,0), Vector{Float32}(undef,0), f, Z->d(f)(Z), learning, λ)
end

mutable struct Mind
    layers::Vector{Layer}
end

function interconnect(layers::Vector{Layer})
    for (l_out, l_in) in zip(layers[2:end], layers[1:end-1])
        if typeof(l_out) == HiddenLayer
            l_out.weights = randn(l_out.nodes, l_in.nodes) / √l_in.nodes
            l_out.biases = randn(l_out.nodes)
        elseif typeof(l_out) == ConvolutionalLayer
            l_out.weights = randn(l_out.depth), l_out.filterx*l_out.filtery / √(l_out.filterx*l_out.filtery)
            l_out.biases = randn(l_out.nodes)
        elseif typeof(l_out) == OutputLayer
            l_out.weights = randn(l_out.nodes, l_in.nodes) / √l_in.nodes
            l_out.biases = randn(l_out.nodes)
            return layers
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

cross_entropy(P::Matrix{Float32}, Y::Matrix{Float32}) = -sum(Y .* log.(P .+ eps())) / size(Y,2)

function backprop!(mind::Mind, X::Matrix{Float32}, Y::Matrix{Float32}, l=1)
    if typeof(mind.layers[l]) == InputLayer 
        return backprop!(mind, X, Y, l+1)
    elseif typeof(mind.layers[l]) == OutputLayer 
        Z = mind.layers[l].f(mind.layers[l].weights*X .+ mind.layers[l].biases)
        δ = mind.layers[l].df(Z, Y)
        dZ = 1
    else
        Z = mind.layers[l].f(mind.layers[l].weights*X .+ mind.layers[l].biases)
        δ = backprop!(mind, Z, Y, l+1)
        dZ = mind.layers[l].df(Z)
    end
    ∂C = mind.layers[l].weights' * (δ .* dZ)
    if mind.layers[l].learning
        mind.layers[l].biases .-= mind.layers[l].λ * sum(δ, dims=2)
        mind.layers[l].weights .-=  mind.layers[l].λ * (δ * X')
    end
    return ∂C
end

function predict(mind::Mind, X::Matrix{Float32}, l=1)
    if typeof(mind.layers[l]) == InputLayer 
        return predict(mind, X, l+1)
    elseif typeof(mind.layers[l]) == OutputLayer 
        return mind.layers[l].f(mind.layers[l].weights*X .+ mind.layers[l].biases)
    elseif typeof(mind.layers[l]) == ConvolutionalLayer 
        convolutes = zeros(Float32, mind.layers[l].nodes, size(X,2))
        for raster_y ∈ 0:mind.layers[l].imagey-mind.layers[l].filtery
            y_coords = raster_y:raster_y+mind.layers[l].filtery-1
            for raster_x ∈ 1:mind.layers[l].imagex-mind.layers[l].filterx+1
                x_coords = raster_x:raster_x+mind.layers[l].filterx-1
                image_indices = [x + y * mind.layers[l].imagex for x ∈ x_coords for y ∈ y_coords]
                output_indices = [raster_x + 
                                    (raster_y + d  * (mind.layers[l].imagey - mind.layers[l].filtery + 1)) 
                                    * (mind.layers[l].imagex - mind.layers[l].filterx + 1) for 
                                 d ∈ 0:mind.layers[l].depth-1]
                convolutes[output_indices,:] .+= (mind.layers[l].weights .* X[image_indices,:]  .+ mind.layers[l].biases)
            end
        end
        Z = mind.layers[l].f(convolutes)
        return predict(mind, Z, l+1)
    else
        Z = mind.layers[l].f(mind.layers[l].weights*X .+ mind.layers[l].biases)
        return predict(mind, Z, l+1)
    end
end

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
                X2::Matrix{Float32}, Y2::Matrix{Float32}; cycles=1, batchsize=128)
    training_skorz = []
    test_skorz = []
    for _ ∈ 1:cycles
        for indices ∈ batch(size(X,2), batchsize)
            backprop!(mind, X[:, indices], Y[:, indices], 1)
        end
        push!(training_skorz, mind.layers[end].score(predict(mind, X), Y))
        push!(test_skorz, mind.layers[end].score(predict(mind, X2), Y2))
    end
    return training_skorz, test_skorz
end

function learn!(mind::Mind, X::Matrix{Float32}, Y::Matrix{Float32}; cycles=1, batchsize=128)
    training_skorz = []
    for _ ∈ 1:cycles
        for indices ∈ batch(size(X,2), batchsize)
            backprop!(mind, X[:, indices], Y[:, indices], 1)
        end
        push!(training_skorz, mind.layers[end].score(predict(mind, X), Y))
    end
    return training_skorz
end

end # module
