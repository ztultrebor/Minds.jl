module Minds

export Mind, learn!, predict, educate!

mutable struct Mind
    layers::Vector{Int}
    weights::Vector{Matrix{Float32}}
    biases::Vector{Vector{Float32}}
    λ::Float32
    a::Function
    f::Function
    da::Function
end

function Mind(layers; λ=0.01, a=relu, f=softmax)
    ws = Vector{Matrix{Float32}}(undef, length(layers)-1)
    bs = Vector{Vector{Float32}}(undef, length(layers)-1)
    for (i, (nin, nout)) in enumerate(zip(layers[1:end-1], layers[2:end]))
        ws[i] = randn(nout, nin) / √nin
        bs[i] = randn(nout)
    end
    return Mind(layers, ws, bs, λ, a, f, d(a))
end

relu(X) = max.(X, 0)

σ(X) = 1 ./ (1 .+ exp.(X))

function d(f::Function)
    if f==relu
        df(Z) = max.(sign.(Z), 0)
    elseif f == σ
        df(Z) = Z .* (1 .- Z)
    end
    return dF
end

function softmax(X)
    E = exp.(X)
    M = sum(E, dims=1)
    return E ./ M
end

function backprop!(mind::Mind, X::Matrix{Float32}, Y::Matrix{Float32}, l=1)
    if l == length(mind.layers) - 1
        Z = softmax(mind.weights[l]*X .+ mind.biases[l])
        δ = (Z .- Y) / size(Z,2)
    else
        Z = relu(mind.weights[l]*X .+ mind.biases[l])
        δ = backprop!(mind, Z, Y, l+1)
    end
    dZ = df(Z)
    ∂C = mind.weights[l]' * (δ .* dZ)
    mind.biases[l] .-= mind.λ * sum(δ, dims=2)
    mind.weights[l] .-=  mind.λ * (δ * X')
    return ∂C
end

function thoughtless_backprop(mind::Mind, X::Matrix{Float32}, Y::Matrix{Float32}, l=1)
    if l == length(mind.layers) - 1
        Z = softmax(mind.weights[l]*X .+ mind.biases[l])
        δ = (Z .- Y) / size(Z,2)
    else
        Z = relu(mind.weights[l]*X .+ mind.biases[l])
        δ = backprop!(mind, Z, Y, l+1)
    end
    dZ = df(Z)
    ∂C = mind.weights[l]' * (δ .* dZ)
    return ∂C
end

function teaching_backprop!(mind::Mind, teacher::Mind, X::Matrix{Float32}, Y::Matrix{Float32}, l=1)
    if l == length(mind.layers)
        return thoughtless_backprop(teacher, X, Y, 1)
    else
        Z = relu(mind.weights[l]*X .+ mind.biases[l])
        δ = teaching_backprop!(mind, teacher, Z, Y, l+1)
    end
    dZ = df(Z)
    ∂C = mind.weights[l]' * (δ .* dZ)
    mind.biases[l] .-= mind.λ * sum(δ, dims=2)
    mind.weights[l] .-=  mind.λ * (δ * X')
    return ∂C
end


function predict(mind::Mind, X::Matrix{Float32}, l=1)
    if l == length(mind.layers)-1
        return softmax(mind.weights[l]*X .+ mind.biases[l])
    else
        Z = relu(mind.weights[l]*X .+ mind.biases[l])
        return predict(mind, Z, l+1)
    end
end

score(P::Matrix{Float32}, Y::Matrix{Float32}) = -sum(Y .* log.(P .+ eps())) / size(Y,2)

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
        push!(training_skorz, score(predict(mind, X), Y))
        push!(test_skorz, score(predict(mind, X2), Y2))
    end
    return training_skorz, test_skorz
end

function educate!(mind::Mind, teacher::Mind, X::Matrix{Float32}, Y::Matrix{Float32},  
    X2::Matrix{Float32}, Y2::Matrix{Float32}, cycles::Int)
    training_skorz = []
    test_skorz = []
    for cycle ∈ 1:cycles
        for randindices = batch(size(X,2), 128)
            x = X[:, randindices]
            y = Y[:, randindices]
            teaching_backprop!(mind, teacher, x, y, 1)
        end
        push!(training_skorz, score(predict(teacher, predict(mind, X)), Y))
        push!(test_skorz, score(predict(teacher, predict(mind, X2)), Y2))
    end
    return training_skorz, test_skorz
end

end # module
