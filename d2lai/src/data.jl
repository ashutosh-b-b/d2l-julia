using StatsBase 
using Downloads 

abstract type AbstractData end

function Base.show(io::IO, mime::MIME"text/plain", data::T) where T <: AbstractData 
    _typename = Base.typename(T).wrapper
    println(io, "Data object of type $(_typename)")
end

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
    function FashionMNISTData(; batchsize = 64, resize = nothing, flatten = false)
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
        args = (batchsize = batchsize, flatten = flatten, resize = resize)
        new{typeof(t), typeof(v), typeof(l), typeof(args)}(t, v, l, args)
    end
end

function get_dataloader(data::FashionMNISTData; train = true, flatten = data.args.flatten)
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


function _download(dataset::AbstractString)
    download_dir = mktempdir()
    file_path = Downloads.download(d2lai.DATA_URL*dataset, joinpath(download_dir, dataset))
end

function _download(::Type{TimeMachine}) 
    file_path = _download("timemachine.txt")
    s = open(file_path, "r") do f
        read(f, String)
    end
    return s
end

function _preprocess(::Type{TimeMachine}, raw_text)
    text = replace(raw_text, r"[^A-Za-z]+" => " ") |> lowercase
end 

function _tokenize(::Type{TimeMachine}, text)
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
        tokens = reduce(vcat, tokens)
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

function build(T::Type{TimeMachine}, raw_text, vocab = nothing)
    tokens = _tokenize(T, _preprocess(T, raw_text))
    if isnothing(vocab)
        vocab = Vocab(; tokens)
    end
    corpus = [vocab[token] for token in tokens]
    return corpus, vocab
end

function d2lai.TimeMachine(batchsize::Int, num_steps::Int, num_train = 10000, num_val = 5000)
    corpus, vocab = d2lai.build(TimeMachine, d2lai._download(TimeMachine))
    array = reduce(hcat, [corpus[i:i+num_steps] for i in 1:(length(corpus) - num_steps)])
    X,y = array[1:end-1, :], array[2:end, :]
    d2lai.TimeMachine(X, y, vocab, corpus, (batchsize = batchsize, num_steps = num_steps, num_train=num_train, num_val = num_val))
end

function get_dataloader(data::d2lai.TimeMachine; train = true)
    idxs = train ? (1:data.args.num_train) : (data.args.num_train+1):(data.args.num_train+data.args.num_val)
    return Flux.DataLoader((Array{Int}(Flux.onehotbatch(data.X[:, idxs], 1:length(data.vocab))), data.y[:, idxs]), shuffle = train, batchsize = data.args.batchsize)
end

## MTFraEng 

struct MTFraEng{S, T, A, AG} <: AbstractData 
    src_vocab::S 
    tgt_vocab::T
    arrays::A 
    args::AG
end

function _extract(zip_path::AbstractString)
    run(`$(p7zip()) x $zip_path -o$(dirname(zip_path)) -y -bso0 -bse0`)
    return dirname(zip_path)
end

function _download(::Type{MTFraEng})
    folder = _extract(d2lai._download("fra-eng.zip"))
    s = open(joinpath(folder, "fra-eng/", "fra.txt"), "r") do f
        read(f, String)
    end
    return s
end



function _preprocess(::Type{MTFraEng}, text::AbstractString)
    # Replace non-breaking space with space
    text = replace(text, '\u202f' => ' ')
    # text = replace(text, '\xa0' => ' ')
    
    # Insert space between words and punctuation marks
    out = Char[]
    for (i, char) in enumerate(lowercase(text))
        if i > 1 && _no_space(char, text[prevind(text, i)])
            push!(out, ' ')
        end
        push!(out, char)
    end
    return String(out)
end

# Helper function
function _no_space(char::Char, prev_char::Char)
    char in (',', '.', '!', '?') && prev_char != ' '
end

function _tokenize(::Type{MTFraEng}, text; max_examples = nothing)
    src = []; tgt = []
    split_text = split(text, '\n')
    max_examples = isnothing(max_examples) ? length(split_text) : max_examples
    vec = map(enumerate(split_text), 1:max_examples) do (i, line), _
        parts = split(line, '\t')
        if length(parts) == 2 
            return collect([t for t in split("$(parts[1]) <eos>", " ")]), collect([t for t in split("$(parts[2]) <eos>", " ")])
        end
    end
    vec = filter(!isnothing, vec)
    return first.(vec), last.(vec)
end


function show_list_len_pair_hist(labels, xlabel, ylabel, x, y)
    histogram(length.(x), label = labels[1], xlabel = xlabel, ylabel = ylabel, bins = 5:5:50)
    histogram!(length.(y), label = labels[2], bins = 5:5:50, alpha = 0.5, )
end


function _build_array(::Type{MTFraEng},sentences, vocab, num_steps; is_tgt = false)
    pad_or_trim = (seq, t) -> length(seq) > t ?  seq[1:t] : vcat(seq, fill("<pad>", t - length(seq)))
    sentences = map(s -> pad_or_trim(s, num_steps), sentences)
    if is_tgt 
        sentences = map(s -> vcat(["<bos>"], s), sentences)
    end
    if isnothing(vocab)
        vocab = Vocab(; tokens = sentences, min_freq = 2)
    end
    array = [vocab[s] for s in sentences]
    valid_len = map(a -> sum(a .!= vocab["<pad>"]), array)
    return reduce(hcat, array), vocab, valid_len
end

function _build_arrays(::Type{MTFraEng}, raw_text, num_steps, src_vocab = nothing, tgt_vocab = nothing)
    src, tgt = _tokenize(MTFraEng, _preprocess(MTFraEng, raw_text))
    src_array, src_vocab, src_valid_len = _build_array(MTFraEng, src, src_vocab, num_steps)
    tgt_array, tgt_vocab, _ = _build_array(MTFraEng, tgt, tgt_vocab, num_steps, is_tgt = true)
    return (src_array, tgt_array[1:end-1, :], src_valid_len, tgt_array[2:end, :]),
            src_vocab, tgt_vocab
end



function MTFraEng(batchsize::Int64, num_steps::Int64=9; num_train=512, num_val=128)
    raw_text = _download(MTFraEng)
    arrays, src_vocab, tgt_vocab = _build_arrays(MTFraEng, raw_text, num_steps)
    args = (; batchsize, num_steps, num_train, num_val)
    MTFraEng(src_vocab, tgt_vocab, arrays, args)
end


function get_dataloader(data::MTFraEng; train = true)
    idxs = train ? (1:data.args.num_train) : (data.args.num_train+1):(data.args.num_train+data.args.num_val)
    # converts to one hot first 
    # conversion is required because unlike the pytorch implementation, this cannot be part of the inference, due to Zygote diff errors.
    src_arr = data.arrays[1][:, idxs]
    decoder_arr = data.arrays[2][:, idxs]
    labels = data.arrays[4][:, idxs]
    src_valid_len = data.arrays[3][idxs]
    # constructs the dataloader now
    Flux.DataLoader((src_arr, decoder_arr, src_valid_len, labels), shuffle = train, batchsize = data.args.batchsize)
end

function build(data::MTFraEng, src_sentences, tgt_sentences)
    raw_text = join([src * "\t" * tgt for (src, tgt) in zip(src_sentences, tgt_sentences)], "\n")
    arrays, _ = _build_arrays(MTFraEng, raw_text, data.args.num_steps, data.src_vocab, data.tgt_vocab)
    arrays
end