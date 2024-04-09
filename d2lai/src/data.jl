abstract type AbstractData end
struct SyntheticRegressionData <: AbstractData
    X::AbstractArray 
    y::AbstractArray 
    args::NamedTuple
    function SyntheticRegressionData(w, b, noise = 0.01, num_train = 1000, num_val = 1000, batchsize = 32)
        args = (noise = noise, num_train = num_train, num_val = num_val, batchsize = batchsize)
        n = args.num_train + args.num_val 
        X = randn(length(w), n)
        y = w*X .+ b .+ randn(1, n).*noise
        new(X, y, args)
    end
end

function get_dataloader(data::AbstractData; train = true)
        indices = train ? Random.shuffle(1:data.args.num_train) : (data.args.num_train+1):(data.args.num_train+data.args.num_val)
        partitioned_indices = collect(Iterators.partition(indices, data.args.batchsize))
        data = map(partitioned_indices) do idx 
            data.X[:, idx], data.y[:, idx]
        end
        data
end

train_dataloader(data::AbstractData) = get_dataloader(data; train = true)
val_dataloader(data::AbstractData) = get_dataloader(data; train = false)

struct FashionMNISTData{T,V,L,A} <: AbstractData 
    train::T
    val::V
    labels::L
    args::A
    function FashionMNISTData(; batchsize = 64)
        dataset = MLDatasets.FashionMNIST
        t = dataset(:train)[:]
        v = dataset(:test)[:]
        l = dataset().metadata["class_names"]
        args = (batchsize = batchsize,)
        new{typeof(t), typeof(v), typeof(l), typeof(args)}(t, v, l, args)
    end
end

function get_dataloader(data::FashionMNISTData; train = true)
    d = train ? data.train : data.val 
    return Flux.DataLoader((Flux.flatten(d[1]), d[2]); batchsize = data.args.batchsize, shuffle = train)
end