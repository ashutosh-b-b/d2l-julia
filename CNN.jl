# 7.2 Convolutions for Images
function corr2d(X::AbstractArray, K::AbstractArray)
    kh, kw = size(K)
    y = Zygote.Buffer(zeros(size(X,1) - size(K,1) + 1, size(X,2) - size(K,2) + 1, size(X)[3:end]...))
    for i in 1:(size(X, 1) - size(K, 1) + 1)
        for j in 1:(size(X, 2) - size(K, 2) + 1)
            y[i,j] = sum(X[i:(i+kh-1), j:(j+kw-1)].*K)
        end
    end
    copy(y)
end

X = [0 1 2; 3 4 5; 6 7 8];
K = [0 1; 2 3]
@show corr2d(X, K)

struct Conv2d{UT, KT} <: AbstractModel 
    K::KT
    U::UT
    function Conv2d(kernel_size; bias = true)
        K = rand(kernel_size...)
        U = bias ? zeros(1) : nothing
        @info typeof(K)
        new{typeof(U), typeof(K)}(K, U)
    end
end

Functors.@functor Conv2d

(model::Conv2d{<:AbstractArray, <:AbstractArray})(X::AbstractArray) = U .+ corr2d(X, model.K)
(model::Conv2d{Nothing})(X::AbstractArray) = corr2d(X, model.K)

X = ones(6,8)
X[:, 3:6] .= 0
X
K = [1.0 -1.0]
Y = corr2d(X, K)
corr2d(X', K)

conv2d = Conv2d((1,2); bias = false)
X = reshape(X, 6, 8, 1, 1)
Y = reshape(Y, 6,7,1)
lr = 3e-2

for i in 1:1000
    ps = Flux.params(conv2d)
    gs = gradient(ps) do 
        Y_pred = conv2d(X)
        l = Flux.mse(Y, Y_pred)
    end
    Y_pred = conv2d(X)
    l = Flux.mse(Y, Y_pred)
    @info "Loss : $l"
    conv2d.K .-= lr*gs[ps[1]]
end

@info conv2d.K

# 7.3 Padding and Stride 
using Flux 

function comp_conv(conv2d, X)
    X = reshape(X, size(X)..., 1, 1)
    Y = conv2d(X)
    return Y[:,:,1,1]
end

conv2d = Flux.Conv((3, 3), 1 => 1, bias = false, pad = 1)

@info comp_conv(conv2d, rand(8,8)) |> size

conv2d = Flux.Conv((5,3), 1 => 1, bias = false, pad = (2,1))

@info comp_conv(conv2d, rand(8,8)) |> size

conv2d = Flux.Conv((3, 3), 1 => 1, bias = false, pad = 1, stride = 2)

@info comp_conv(conv2d, rand(8,8)) |> size

conv2d = Flux.Conv((3, 3), 1 => 1, bias = false, pad = (0, 1), stride = (3,4))

@info comp_conv(conv2d, rand(8,8)) |> size

## 7.4 Multiple Input Multiple Output 
function corr2d_multi(X, K)
    mapreduce(corr2d, .+, eachslice(X, dims = length(size(X))), eachslice(K, dims = length(size(K))))
end

X  = cat([0 1 2; 3 4 5; 6 7 8] ,[1 2 3; 4 5 6; 7 8 9]; dims = 3)
K = cat([0 1; 2 3], [1 2; 3 4]; dims = 3)

corr2d_multi(X, K)

function corr2d_multi_in_out(X, K)
    stack(map(k -> corr2d_multi(X, K) , eachslice(K, dims = 4)), dims = 3)
end




function pool2d(X, pool_size; mode = maximum)
    Y = zeros(size(X) .- pool_size .+ 1)
    for i in 1:(size(X, 1) - pool_size[1] + 1)
        for j in 1:(size(X, 2) - pool_size[2] + 1)
            Y[i, j] = mode(X[i:i+pool_size[1] - 1, j:j+pool_size[2] - 1])
        end
    end
    Y
end

X = [0 1 2; 3 4 5; 6 7 8]

pool2d(X, (2,2))
pool2d(X, (2,2); mode = mean)

X = reshape(collect(0:1:15), (4,4,1,1))

MaxPool((3,3))(X)

MaxPool((3,3), pad = 1, stride = 2)(X)
