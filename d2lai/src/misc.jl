function corr2d(X::AbstractArray, K::AbstractArray)
    Y = zeros(size(X) .- size(K) .+ 1)
    kh, kw = size(K)
    for i in 1:size(Y, 1)
        for j in 1:size(Y, 2)
            Y[i, j] = sum(X[i:(i+kh-1), j:j+kw-1] .* K)
        end
    end
    Y
end

function to_gpu(model)
    Flux.fmap(model) do x
        x isa AbstractArray || return x
        Flux.gpu(x)
    end
end