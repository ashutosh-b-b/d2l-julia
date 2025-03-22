using StatsBase 
using Downloads 

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
    function FashionMNISTData(; batchsize = 64, resize = nothing)
        dataset = MLDatasets.FashionMNIST
        t = dataset(:train)[:]
        v = dataset(:test)[:]
        t = if isnothing(resize) 
            t 
        else
            features_resize = imresize(t.features, resize)
            (features = features_resize, targets = t.targets)
        end

        v = if isnothing(resize) 
            v 
        else
            features_resize = imresize(v.features, resize)
            (features = features_resize, targets = v.targets)
        end
        l = dataset().metadata["class_names"]
        args = (batchsize = batchsize,)
        new{typeof(t), typeof(v), typeof(l), typeof(args)}(t, v, l, args)
    end
end

function get_dataloader(data::FashionMNISTData; train = true, flatten = false)
    d = train ? data.train : data.val 
    if flatten 
        Flux.DataLoader((Flux.flatten(d[1]), d[2]); batchsize = data.args.batchsize, shuffle = train)
    else
        d_reshaped = reshape(d[1], size(d[1])[1], size(d[1])[2], 1, :)
        Flux.DataLoader((d_reshaped, d[2]); batchsize = data.args.batchsize, shuffle = train)
    end
end

################## Time Machine Dataset #########################
mutable struct TimeMachine{X, Y, V, C, A} <: AbstractData
    X::X 
    y::Y
    vocab::V
    corpus::C
    args::A
end

function _download()
    download_dir = mktempdir()
    file_path = Downloads.download(d2lai.DATA_URL*"timemachine.txt", joinpath(download_dir, "timemachine.txt"))
    s = open(file_path, "r") do f
        read(f, String)
    end
    return s
end

function _preprocess(raw_text)
    text = replace(raw_text, r"[^A-Za-z]+" => " ") |> lowercase
end 

function _tokenize(text)
    return string.(collect(text))
end 

struct Vocab{TF, IT, TI}
    token_freqs::TF 
    idx_to_token::IT 
    token_to_idx::TI
end 

function Vocab(; tokens = [], min_freq = 0, reserved_tokens = [])
    # Flatten a 2D list if needed
    if !isempty(tokens) && tokens[1] isa Vector
        tokens = vcat(tokens...)
    end
    
    # Count token frequencies
    counter = countmap(tokens)
    token_freqs = sort(collect(counter), by=x->x[2], rev=true)
    # The list of unique tokens
    idx_to_token = sort(vcat(["<unk>"], reserved_tokens, 
        [(string(token)) for (token, freq) in token_freqs if freq >= min_freq]))
    
    # Token to index mapping
    token_to_idx = Dict(token => idx for (idx, token) in enumerate(idx_to_token))

    Vocab(token_freqs, idx_to_token, token_to_idx)

end

Base.length(v::Vocab) = length(v.idx_to_token)
unk(v::Vocab) = v.token_to_idx["<unk>"]

function Base.getindex(v::Vocab, tokens)
    if !(typeof(tokens) <: AbstractVector)
        return haskey(v.token_to_idx, tokens) ? v.token_to_idx[string(tokens)] : unk(v)
    else
        return map(t -> Base.getindex(v, t), string.(tokens))
    end
end

to_tokens(v::Vocab, idx::Int) = v.idx_to_token[idx]
to_tokens(v::Vocab, indices::AbstractVector{<:Int}) = to_tokens.(Ref(v), indices)

function build(raw_text, vocab = nothing)
    tokens = _tokenize(_preprocess(raw_text))
    if isnothing(vocab)
        vocab = Vocab(; tokens)
    end
    corpus = [vocab[token] for token in tokens]
    return corpus, vocab
end

function d2lai.TimeMachine(batchsize::Int, num_steps::Int, num_train = 10000, num_val = 5000)
    corpus, vocab = d2lai.build(d2lai._download())
    array = reduce(hcat, [corpus[i:i+num_steps] for i in 1:(length(corpus) - num_steps)])
    X,y = array[1:end-1, :], array[2:end, :]
    d2lai.TimeMachine(X, y, vocab, corpus, (batchsize = batchsize, num_steps = num_steps, num_train=num_train, num_val = num_val))
end

function get_dataloader(data::d2lai.TimeMachine; train = true)
    idxs = train ? (1:data.args.num_train) : (data.args.num_train+1):(data.args.num_train+data.args.num_val)
    return Flux.DataLoader((Array{Int}(Flux.onehotbatch(data.X[:, idxs], 1:length(data.vocab))), data.y[:, idxs]), shuffle = train, batchsize = data.args.batchsize)
end