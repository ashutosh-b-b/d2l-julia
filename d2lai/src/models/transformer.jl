function predict_step(model::AbstractEncoderDecoder, batch, device, num_steps; save_attention_wts = false)
    batch = batch |> device 
    src, tgt, src_valid_len, _ = batch
    enc_all_outputs = model.encoder(src, src_valid_len)
    dec_state = d2lai.init_state(model.decoder, enc_all_outputs, src_valid_len)
    outputs, attention_weights = [tgt[1:1, 1:end]], []
    for _ in 1:num_steps 
        Y_t = outputs[end]
        Y_t_plus_1, dec_state, attention_wts = model.decoder(Y_t, dec_state; return_attention_wts = save_attention_wts)
        Y_t_plus_1 = softmax(Y_t_plus_1, dims = 1)
        Y_t_plus_1_index = getindex.(argmax(Y_t_plus_1, dims = 1), 1)
        push!(outputs, reshape(Y_t_plus_1_index, 1, :))
        save_attention_wts && push!(attention_weights, attention_wts)
    end
    out = reduce(vcat, outputs)
    if !save_attention_wts
        return out[2:end, :]
    else
        return out[2:end, :], attention_weights
    end
end


function bleu(pred_seq::String, label_seq::String, k::Int)
    """Compute the BLEU score."""
    
    pred_tokens = split(pred_seq)
    label_tokens = split(label_seq)
    len_pred = length(pred_tokens)
    len_label = length(label_tokens)
    
    # Brevity penalty
    score = exp(min(0.0, 1 - len_label / len_pred))
    
    for n in 1:min(k, len_pred)
        num_matches = 0
        label_subs = Dict{String, Int}()
        
        # Build reference n-gram counts
        for i in 1:(len_label - n + 1)
            ngram = join(label_tokens[i:i+n-1], " ")
            label_subs[ngram] = get(label_subs, ngram, 0) + 1
        end
        
        # Match predicted n-grams against reference
        for i in 1:(len_pred - n + 1)
            pred_ngram = join(pred_tokens[i:i+n-1], " ")
            if get(label_subs, pred_ngram, 0) > 0
                num_matches += 1
                label_subs[pred_ngram] -= 1
            end
        end
        
        # Update score with weighted precision
        score *= (num_matches / (len_pred - n + 1))^(0.5^n)
    end
    
    return score
end

function show_heatmaps(matrices, xlabel, ylabel, titles = nothing)
    num_rows, num_cols = size(matrices)[end-1:end]
    layout = (num_rows, num_cols)
    titles = isnothing(titles) ? Iterators.repeated(nothing) : titles
    heatmaps = map(eachslice(matrices, dims = (3,4)), titles) do matrix, title
        isnothing(title) && return heatmap(matrix; xlabel, ylabel)
        return heatmap(matrix; xlabel, ylabel, title)
    end
    plot(heatmaps...; layout)
end

# function _sequence_mask_(X::AbstractArray, valid_len, value::T = 0) where {T}
#     n, q, b = size(X)
#     device = isa(X, CuArray) ? cu : identity
#     key_ids = reshape(device(collect(1:n)), n, 1, 1)

#     if eltype(valid_len) <: AbstractVector
#         # valid_len is Vector of Vectors
#         # Build a (n, q, b) mask with broadcasting
#         valid_mat = zeros(Int, q, b)
#         for j in 1:b
#             valid_mat[:, j] .= valid_len[j]
#         end
#         valid_mat = reshape(device(valid_mat), 1, q, b)
#     else
#         # valid_len is simple vector
#         valid_mat = reshape(device(collect(valid_len)), 1, 1, b)
#     end
    

#     mask = key_ids .<= valid_mat  # shape: (n, q, b)
#     mask_f = T.(mask)
#     return X .* mask_f .+ value .* (1 .- mask_f)
# end

# function masked_softmax(X, valid_lens, value = 0.)
#     if isnothing(valid_lens)
#         return softmax(X, dims = 1)
#     else
#         X_ = _sequence_mask_(X, valid_lens, -1e6)
#         return softmax(X_, dims = 1)
#     end
# end


function sequence_mask(X::AbstractArray, valid_len::AbstractVector, value=0.)
    valid_len = valid_len |> cpu
    device = isa(X, CuArray) ? gpu : cpu
    masks = map(valid_len) do vl
        if !isa(vl, AbstractVector)
            return [i <= vl for i in 1:size(X, 1), j in 1:size(X, 2)]
        else
            return [i <= vl[j] for i in 1:size(X, 1), j in 1:size(X, 2)]
        end
    end
    mask = cat(masks..., dims = 3) |> device
    X.*mask .+ (1 .- mask).*value
end
### Masked Softmax ####
# function _sequence_mask_(X::AbstractArray, valid_len, value::T = 0) where {T}
#     n, q, b = size(X)
#     device = isa(X, CuArray) ? cu : identity
#     key_ids = reshape(device(collect(1:n)), n, 1, 1)

#     if eltype(valid_len) <: AbstractVector
#         # valid_len is Vector of Vectors
#         # Build a (n, q, b) mask with broadcasting
#         valid_mat = zeros(Int, q, b)
#         for j in 1:b
#             valid_mat[:, j] .= valid_len[j]
#         end
#         valid_mat = reshape(device(valid_mat), 1, q, b)
#     else
#         # valid_len is simple vector
#         valid_mat = reshape(device(collect(valid_len)), 1, 1, b)
#     end
    

#     mask = key_ids .<= valid_mat  # shape: (n, q, b)
#     mask_f = T.(mask)
#     return X .* mask_f .+ value .* (1 .- mask_f)
# end

function masked_softmax(X, valid_lens, value = 0.)
    if isnothing(valid_lens)
        return softmax(X, dims = 1)
    else
        X_ = sequence_mask(X, valid_lens, -1e6)
        return softmax(X_, dims = 1)
    end
end

# function sequence_mask(X::AbstractArray, valid_lens, value)
#     device = isa(X, CuArray) ? gpu : cpu
#     Xs = map(enumerate(eachslice(X, dims = 3))) do (i, x )
#         vl = valid_lens[i:i]
#         mask = if length(vl) > 1
#             msk = map(vl) do v 
#                collect(1:size(X,1)) .<= v
#             end
#             reduce(hcat, msk) 
#         else
#             reduce(hcat, fill(collect(1:size(X, 1)) .<= vl, size(X, 2)))
#         end |> device

#         x.*mask .+ (1 .- mask).*value 
#     end
#     cat(Xs..., dims = 3)
#     # Xs = map(eachslice(X, dims = 3), valid_lens) do x_, vl
#     #     x_
#     #     # if isa(vl, AbstractVector)
#     #     #     mask = mapreduce(hcat, vl) do v 
#     #     #         1:size(X,1) .<= v
#     #     #     end |> device
#     #     # else
#     #     #     mask = reduce(hcat, 
#     #     #         fill(1:size(X, 1) .<= vl, size(X, 2))
#     #     #     ) |> device
#     #     # end
#     #     # x_.*mask .+ (1 .- mask).*value |> device
        
#     # end
#     # # cat(Xs, dims = 3)
# end

# function masked_softmax(X::AbstractArray, valid_lens)
#     if isnothing(valid_lens) 
#         return softmax(X, dims = 1)
#     else
#         X_ = sequence_mask(X, valid_lens, -1e6)
#         return softmax(X_, dims = 1)
#     end
# end

# function _sequence_mask_(X::AbstractArray, valid_len, value::T = 0) where {T}
#     n, q, b = size(X)
#     device = isa(X, CuArray) ? cu : identity
#     key_ids = reshape(device(collect(1:n)), n, 1, 1)

#     if eltype(valid_len) <: AbstractVector
#         # valid_len is Vector of Vectors
#         # Build a (n, q, b) mask with broadcasting
#         # valid_mat = zeros(Int, q, b)
#         # for j in 1:b
#         #     valid_mat[:, j] .= valid_len[j]
#         # end
#         valid_mat = Matrix(reduce(hcat, valid_len)')
#         valid_mat = reshape(device(valid_mat), 1, q, b)
#     else
#         # valid_len is simple vector
#         valid_mat = reshape(device(collect(valid_len)), 1, 1, b)
#     end
    

#     mask = key_ids .<= valid_mat  # shape: (n, q, b)
#     mask_f = T.(mask)
#     return X .* mask_f .+ value .* (1 .- mask_f)
# end

# function masked_softmax(X, valid_lens, value = 0.)
#     if isnothing(valid_lens)
#         return softmax(X, dims = 1)
#     else
#         X_ = _sequence_mask_(X, valid_lens, -1e6)
#         return softmax(X_, dims = 1)
#     end
# end


# function sequence_mask(X::AbstractArray{T, 3}, valid_len::AbstractVector{<:Real}, value = 0.) where T
#     v_d = size(X, 1)
#     device = isa(X, CuArray) ? gpu : cpu
#     valid_len_reshape = reshape(valid_len, 1, 1, :)  |> device
#     v_d_reshape = reshape(1:v_d, :, 1, 1) |> device
#     msk = valid_len_reshape .< v_d_reshape
#     return X.* msk.*value + X.*(1 .- msk)
# end


# function sequence_mask(X::AbstractArray{T, 3}, valid_len::AbstractVector{<:AbstractVector}, value = 0.) where T
#     v_d = size(X, 1)
#     device = isa(X, CuArray) ? gpu : cpu
#     valid_len_reshape = reshape(reduce(hcat, valid_len), 1, size(X, 2), :) |> device
#     v_d_reshape = reshape(1:v_d, :, 1, 1) |> device
#     msk = valid_len_reshape .< v_d_reshape
#     return X.* msk.*value + X.*(1 .- msk)
# end

# function sequence_mask(X::AbstractArray{T, 3}, valid_len::AbstractVector{<:Real}, value=0.0) where T
#     v_d = size(X, 1)
#     device_array = X isa CuArray ? CUDA.zeros(Bool, 1, 1, length(valid_len)) : falses(1, 1, length(valid_len))
#     valid_len = convert.(eltype(X), valid_len)  # Match X's type
#     msk = reshape(valid_len, 1, 1, :) .> reshape(0:v_d-1, :, 1, 1)
#     msk = X isa CuArray ? cu(msk) : msk
#     return ifelse(value == 0.0, X .* msk, X .* msk .+ value .* .!msk)
# end

# # # For nested valid_lens (2D as vector of vectors)
# # function sequence_mask(X::AbstractArray{T, 3}, valid_len::AbstractVector{<:AbstractVector}, value=0.0) where T
# #     v_d = size(X, 1)
# #     lens_mat = reduce(hcat, valid_len)
# #     device_array = X isa CuArray ? CUDA.zeros(Bool, 1, size(X, 2), size(X, 3)) : falses(1, size(X, 2), size(X, 3))
# #     lens_mat = convert.(eltype(X), lens_mat)  # Ensure type compatibility
# #     msk = reshape(lens_mat, 1, size(X, 2), :) .> reshape(0:v_d-1, :, 1, 1)
# #     msk = X isa CuArray ? cu(msk) : msk
# #     return ifelse(value == 0.0, X .* msk, X .* msk .+ value .* .!msk)
# # end

# function masked_softmax(X::AbstractArray, valid_len)
#     isnothing(valid_len) && return softmax(X, dims = 1)
#     return sequence_mask(X, valid_len, -1e16)
# end

# function _sequence_mask(X::AbstractArray{T,2}, valid_len::AbstractVector, value=-1e6) where T
#     maxlen = size(X, 2)
#     device = isa(X, CuArray) ? gpu : cpu
#     positions = reshape(collect(1:maxlen),1 , :) |> device 
#     thresholds = reshape(valid_len, :, 1) |> device
#     #  mask = one(T) .- max.(sign.(positions .- thresholds .- eps(T)), zero(T))
#     mask = positions .< thresholds .|> Int64
#     # return mask .* X .+ (one(T) .- mask) .* value # Zygote-compatible non-mutating version
#     return mask .* X .+ (1. .- mask) .* value
# end
# function masked_softmax(X::AbstractArray{T,3}, valid_lens::Union{AbstractVector,Nothing}=nothing) where T
#     """Perform softmax operation by masking elements on the last axis."""
#     X = permutedims(X, (3, 2, 1))
#     result = if valid_lens === nothing
#         softmax(X, dims=3)
#     else
#         shape = size(X)
#         valid_lens = if isa(valid_lens, AbstractVector)
#              repeat(valid_lens, inner=shape[2])
#         else
#             reshape(valid_lens, :)
#         end
#         # Reshape, mask, then reshape back
#         X_reshaped = reshape(X, :, shape[end])
#         X_masked = _sequence_mask(X_reshaped, valid_lens, -1e6)
#         softmax(reshape(X_masked, shape), dims=3)
#     end
#     return permutedims(result, (3, 2, 1))
# end

struct DotProductAttention{D, A}
    dropout::D 
    args::A 
end


function (m::DotProductAttention)(queries, keys, values, valid_len = nothing)
    # keys -> d x num_keys x batch_size
    # queries -> d x num_queries x batch_size
    d = size(queries, 1)
    scores = batched_mul(batched_transpose(keys), queries) ./ sqrt(d)
    # scores -> num_keys x num_queries x batch_size 
    attention_weights = masked_softmax(scores, valid_len)
    # attention_weights -> num_keys x num_queries x batch_size
    return batched_mul(values, m.dropout(attention_weights)), attention_weights
end

struct AdditiveAttention{W, A, D} <: AbstractModel
    weights::W 
    dropout::D 
    args::A 
end

Flux.@layer AdditiveAttention trainable = (weights,)

function AdditiveAttention(k_d, q_d, v_d, num_hiddens::Int64, dropout::Float64; kw...)
    W_k = Dense(k_d => num_hiddens; bias = false)
    W_q = Dense(q_d => num_hiddens; bias = false)
    W_v = Dense(num_hiddens => 1; bias = false)
    
    AdditiveAttention((; W_k, W_q, W_v), Flux.Dropout(dropout), (;))
end

function (m::AdditiveAttention)(queries, keys, values, valid_lens)
    queries = m.weights.W_q(queries) # num_hiddens x num_queries x batch_size
    keys = m.weights.W_k(keys) # num_hiddens x num_keys x batch_size
    features = Flux.unsqueeze(queries, 2) .+ Flux.unsqueeze(keys, 3) # num_hiddens x num_keys x num_queries x batch_size 
    features = tanh.(features)
    scores = m.weights.W_v(features) # 1 x num_keys x num_queries x batch_size 
    scores = dropdims(scores, dims = 1) # num_keys x num_queries x batch_size
    attention_weights = masked_softmax(scores, valid_lens) # num_keys x num_queries x batch_size
    return batched_mul(values, m.dropout(attention_weights)), attention_weights 
    # num_hidden x num_queries x batch_size , num_keys x num_queries x batch_size
end


struct MultiHeadedAttention{W, AT, A} <: AbstractModel
    weights::W 
    attention::AT
    args::A
end

Flux.@layer MultiHeadedAttention trainable = (weights, )
function MultiHeadedAttention(num_hiddens::Int64, num_heads::Int64, dropout::AbstractFloat; bias=false)
    W_q = Dense(num_hiddens, num_hiddens, bias = bias)
    W_k = Dense(num_hiddens, num_hiddens, bias = bias)
    W_v = Dense(num_hiddens, num_hiddens, bias = bias)
    W_o = Dense(num_hiddens, num_hiddens, bias = bias)
    attn = DotProductAttention(Dropout(dropout), (;))
    MultiHeadedAttention((; W_q, W_k, W_v, W_o), attn, (; num_hiddens, num_heads, dropout))
end

function (m::MultiHeadedAttention)(queries, keys, values, valid_lens)
    # queries -> q_d x num_queries x batch_size 
    # keys -> k_d x num_key_val x batch_size 
    # values -> v_d x num_key_val x batch_size
    queries = m.weights.W_q(queries) # num_hiddens x num_queries x batch_size
    queries = transpose_qkv(m, queries) # (num_hiddens / num_heads) x num_queries x (num_heads * batch_size)
    keys = transpose_qkv(m, m.weights.W_k(keys)) # (num_hiddens / num_heads) x num_key_val x (num_heads * batch_size)
    values = transpose_qkv(m, m.weights.W_v(values))# (num_hiddens / num_heads) x num_key_val x (num_heads * batch_size)
    valid_lens = if !isnothing(valid_lens)
        isa(valid_lens, AbstractVector) ? repeat(valid_lens, inner = m.args.num_heads) : repeat(valid_lens, inner = (m.args.num_heads, 1))
        
    end
    scores, attn_wts = m.attention(queries, keys, values, valid_lens) # (num_hiddens / num_heads) x num_queries x (num_heads * batch_size) 
    # attn_wts -> num_key_val x num_queries x batch_size
    output_concat = transpose_output(m, scores) # num_hiddens x num_queries x batch_size
    return m.weights.W_o(output_concat), attn_wts # 
end

function transpose_qkv(m::MultiHeadedAttention, x)
    # x -> num_hiddens x (num_queries or num_key_val) x batch_size 
    num_q_or_key_val = size(x, 2)
    batch_size = size(x, 3)
    x_ = reshape(x, :, m.args.num_heads, num_q_or_key_val, batch_size)
    x_permuted = permutedims(x_, [1, 3, 2, 4]) # (num_hiddens / num_heads) x num_queries x num_heads x batch_size
    return reshape(x_permuted, size(x_permuted)[1], size(x_permuted)[2], :) # (num_hiddens / num_heads) x num_queries x (num_heads * batch_size)
end

function transpose_output(m::MultiHeadedAttention, x)
    x_ = reshape(x, size(x)[1], size(x)[2], m.args.num_heads, :)
    x_permuted = permutedims(x_, [1, 3, 2, 4])
    return reshape(x, :, size(x_permuted)[3], size(x_permuted)[4]) 
end


struct Seq2SeqAttentionDecoder{AT, E, R, D, A} <: AbstractModel
    attention::AT
    embedding::E 
    rnn::R 
    dense::D
    args::A
end
Flux.@layer Seq2SeqAttentionDecoder trainable = (attention, embedding, rnn, dense)
    
function d2lai.init_state(::Seq2SeqAttentionDecoder, enc_all_out, enc_valid_lens)
    outputs, hidden_state = enc_all_out
    return outputs, hidden_state, enc_valid_lens
end

function Seq2SeqAttentionDecoder(vocab_size::Int, embed_size::Int, num_hiddens, num_layers, dropout=0.)
    embedding = Embedding(vocab_size => embed_size)
    rnn = StackedRNN(embed_size + num_hiddens, num_hiddens, num_layers; rnn = Flux.LSTM)
    attention = AdditiveAttention(num_hiddens, num_hiddens, num_hiddens, num_hiddens, dropout)
    dense = Dense(num_hiddens, vocab_size) 
    args = (; vocab_size, embed_size, num_hiddens, num_layers)
    Seq2SeqAttentionDecoder(attention, embedding, rnn, dense, args)
end

function (m::Seq2SeqAttentionDecoder)(X, state; return_attention_wts = false)
    enc_outputs, hidden_state, enc_valid_lens = state
    embeds = m.embedding(X) # num_embeds x num_steps x batch_size 
    outs = map(eachslice(embeds; dims = 2)) do x 
        query = hidden_state[end] # num_hiddens x batch_size 
        query = Flux.unsqueeze(query, dims = 2) # num_hiddens x 1 x batch_size (num_queries = 1)
        context, attention_wts = m.attention(query, enc_outputs, enc_outputs, enc_valid_lens) # num_hiddens x 1 x batch_size 
        embs_and_context = vcat(Flux.unsqueeze(x, dims = 2), context) # (num_embeds + num_hiddem x 1 x batch_size)
        out, hidden_state = m.rnn(embs_and_context, hidden_state)
        out, attention_wts
    end  
    outputs = first.(outs)
    attention_wts = last.(outs)
    outputs_cat = cat(outputs..., dims = 2)
    out_dense = m.dense(outputs_cat)
    if !return_attention_wts 
        return out_dense, (enc_outputs, hidden_state, enc_valid_lens)
    else
        return out_dense,  (enc_outputs, hidden_state, enc_valid_lens), attention_wts
    end
end