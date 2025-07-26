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


function read_data_bananas(extracted_folder; train = true)
    folder = train ? "bananas_train" : "bananas_val"
    folder_path = joinpath(extracted_folder, "banana-detection", folder)
    df = DataFrame(CSV.File(joinpath(folder_path, "label.csv")))
    img_names = df[!, 1]
    targets = df[!, 2:end] |> Array 
    targets = permutedims(targets, (2, 1))
    images = map(img_names) do img_name 
        img = Images.load(joinpath(folder_path, "images", img_name))
        img = Image(img)
        img_tensor = apply(ImageToTensor(), img) |> itemdata
        img_tensor = permutedims(img_tensor, (2,1,3))
    end
    
    images = stack(images; dims = 4)

    images, Flux.unsqueeze(targets, dims = 1) ./ 256
end

struct BananaDataset{T,V,A} <: AbstractData 
    train_data::T 
    val_data::V 
    args::A
end

function BananaDataset(; batchsize = 32)
    file = d2lai._download("banana-detection.zip")

    extracted_folder = d2lai._extract(file)

    train_data = read_data_bananas(extracted_folder; train = true)
    val_data = read_data_bananas(extracted_folder; train = false)
    args = (; extracted_folder, batchsize)
    BananaDataset(train_data, val_data, args)
end

function get_dataloader(data::BananaDataset; train = true)
    if train
        Flux.DataLoader(data.train_data; batchsize = data.args.batchsize, shuffle = true)
    else
        Flux.DataLoader(data.val_data; batchsize = data.args.batchsize)
    end
end

struct VOCSegDataSet{T, V, A} <: d2lai.AbstractData 
    train::T 
    val::V 
    args::A
end

__filter_size(img, sz) = size(img, 1) >= sz[1] && size(img, 2) >= sz[2]


function read_voc_images(extracted_folder; train = true)
    txt_file =  train ? "train.txt" : "val.txt"
    voc_dir = joinpath(extracted_folder, "VOCdevkit/VOC2012")
    txt_fname = joinpath(voc_dir, "ImageSets", "Segmentation", txt_file)
    lines = readlines(txt_fname)
    feature_imgs = map(lines) do img_name
        img = Images.load(joinpath(voc_dir, "JPEGImages", "$img_name.jpg"))
    end 
    labels = map(lines) do img_name
        img = Images.load(joinpath(voc_dir, "SegmentationClass", "$img_name.png"))
    end 
    feature_imgs, labels
end

function voc_colormap2label()
    colormap2label = fill(0, 256^3)  # use -1 for unknown labels
    for (i, cmap) in enumerate(VOC_COLORMAP)
        r, g, b = cmap
        index = (r * 256 + g) * 256 + b + 1  # still 1-based indexing in Julia
        colormap2label[index] = i - 1  # shift class index: 0 to 20
    end
    return colormap2label
end

# function voc_label_indices(colormap, colormap2label)
#     h, w = size(colormap, 1), size(colormap, 2)
#     idx = Array{Int}(undef, h, w)

#     @inbounds for j in 1:w, i in 1:h
#         r = colormap[i, j, 1]
#         g = colormap[i, j, 2]
#         b = colormap[i, j, 3]
#         index = (r * 256 + g) * 256 + b + 1
#         idx[i, j] = colormap2label[index]
#     end

#     return idx
# end
function voc_label_indices(colormap, colormap2label)
    h, w = size(colormap, 1), size(colormap, 2)
    idx = Array{Int}(undef, h, w)

    @inbounds for j in 1:w, i in 1:h
        r = colormap[i, j, 1]
        g = colormap[i, j, 2]
        b = colormap[i, j, 3]
        index = (r * 256 + g) * 256 + b + 1
        idx[i, j] = colormap2label[index]  # now ranges from 0–20 or -1
    end

    return idx
end

function voc_rand_corp(feature, label, ht, width)
    tfm = DataAugmentation.compose(RandomCrop((ht, width)), ImageToTensor())
    randstate = DataAugmentation.getrandstate(tfm)
    feature_ = apply(tfm, Image(feature); randstate) |> itemdata |> collect
    label_ = apply(tfm, Image(label); randstate) |> itemdata |> collect
    feature_, label_
end
function VOCSegDataSet(crop_size; batchsize = 64)
    file = d2lai._download("VOCtrainval_11-May-2012.tar")
    extracted_folder = d2lai._extract(file)
    train_features, train_labels = read_voc_images(extracted_folder; train = true)
    val_features, val_labels = read_voc_images(extracted_folder; train = false)

    colormap2label = voc_colormap2label()
    
    train_features = filter(f -> __filter_size(f, crop_size), train_features)
    train_labels = filter(f -> __filter_size(f, crop_size), train_labels)
    
    val_features = filter(f -> __filter_size(f, crop_size), val_features)
    val_labels = filter(f -> __filter_size(f, crop_size), val_labels)
    
    corped_train = voc_rand_corp.(train_features, train_labels, Ref(crop_size[1]), Ref(crop_size[2]))
    corped_val = voc_rand_corp.(val_features, val_labels, Ref(crop_size[1]), Ref(crop_size[2]))
    
    train_features, train_labels = first.(corped_train), last.(corped_train)
    val_features, val_labels = first.(corped_val), last.(corped_val)

    train_labels = map(l -> Int.(l .* 255), train_labels)
    val_labels = map(l -> Int.(l .* 255), val_labels)

    train_labels = voc_label_indices.(train_labels, Ref(colormap2label))
    val_labels = voc_label_indices.(val_labels, Ref(colormap2label))

    train_features, train_labels = stack(train_features; dims = 4), stack(train_labels; dims = 3)
    val_features, val_labels = stack(val_features; dims = 4), stack(val_labels; dims = 3)

    VOCSegDataSet(
        (train_features, train_labels),
        (val_features, val_labels),
        (; colormap2label, crop_size, batchsize)
    )
end


function load_data_voc(data::VOCSegDataSet)
    train_iter =  get_dataloader(data)
    test_iter = get_dataloader(data; train = false)
    return train_iter, test_iter
end

VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

#@save
VOC_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow",
               "diningtable", "dog", "horse", "motorbike", "person",
               "potted plant", "sheep", "sofa", "train", "tv/monitor"]
function d2lai.get_dataloader(data::VOCSegDataSet; train = true)
    if train 
        return Flux.DataLoader(data.train; shuffle = true, batchsize = data.args.batchsize)
    else
        return Flux.DataLoader(data.val; shuffle = false, batchsize = data.args.batchsize)
    end
end