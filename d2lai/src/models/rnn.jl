
struct RNNLMScratch{R, W, B, A} <: d2lai.AbstractRNNClassifier
    rnn::R
    Wq::W 
    bq::B
    args::A
end

Flux.@layer RNNLMScratch trainable = (rnn, Wq, bq)

function RNNLMScratch(rnn, vocab_size) 
    Wq = randn(vocab_size, rnn.args.num_hiddens)*rnn.args.sigma 
    bq = zeros(vocab_size)
    RNNLMScratch(rnn, Wq, bq, (vocab_size=vocab_size,))
end

function (rnnlm::RNNLMScratch)(x, state = nothing)
    output, _ = rnnlm.rnn(x, state)
    output_layer(rnnlm, output)
end

function output_layer(m::RNNLMScratch, x)
    outs = map(eachslice(x, dims =2)) do x_ 
        m.Wq*x_ .+ m.bq
    end
    outs = stack(outs)
    return permutedims(outs, [1, 3, 2])
end



struct RNNModelConcise{N,R,A} <: AbstractRNNClassifier 
    net::N 
    rnn::R 
    args::A
end

Flux.@layer RNNModelConcise trainable = (net, rnn)
function RNNModelConcise(rnn, num_hiddens::Int, vocab_size::Int)
    net = Dense(num_hiddens => vocab_size)
    return RNNModelConcise(net, rnn, (num_hiddens = num_hiddens, vocab_size = vocab_size))
end

function output_layer(m::RNNModelConcise, out)
    m.net(out)
end

function (m::RNNModelConcise)(x)
    out = m.rnn(x)[1]
    return output_layer(m, out)
end


struct RNNScratch{Wx, Wh, Bh, A} <: AbstractModel 
    Whx::Wx
    Whh::Wh
    b_h::Bh
    args::A
end
Flux.@layer RNNScratch trainable = (Whx, Whh, b_h)

function RNNScratch(num_inputs::Int, num_hiddens::Int; sigma = 0.01)
    Whx = randn(num_hiddens, num_inputs).*sigma 
    Whh = randn(num_hiddens, num_hiddens).*sigma 
    b_h = zeros(num_hiddens)
    RNNScratch(Whx, Whh, b_h, (num_inputs = num_inputs, num_hiddens = num_hiddens, sigma = sigma))
end

function (rnn::RNNScratch)(x::AbstractArray, state = nothing)
    batchsize = size(x, 3)
    device = isa(x, CuArray) ? gpu : cpu
    state = if isnothing(state)
        zeros(rnn.args.num_hiddens, size(x, 3))
    else
        state
    end |> device
    outputs = map(eachslice(x, dims = 2)) do x_ 
        state = tanh.(rnn.Whx*x_ + rnn.Whh*state .+ rnn.b_h)
        return state
    end
    outputs_cat = stack(outputs)
    return permutedims(outputs_cat, [1, 3, 2]), state  # num_hiddens x num_steps x batchsize, num_hiddens x batchsize
end

struct LSTMScratch{W, A} <: AbstractModel 
    weights::W 
    args::A
end 
Flux.@layer LSTMScratch trainable = (weights,)

function LSTMScratch(num_inputs::Int, num_hiddens::Int; sigma = 0.1)
    init_weights() = randn(num_hiddens, num_inputs).*sigma , randn(num_hiddens, num_hiddens).*sigma, zeros(num_hiddens)
    W_ix, W_ih, b_i = init_weights()
    W_fx, W_fh, b_f = init_weights()
    W_cx, W_ch, b_c = init_weights()
    W_ox, W_oh, b_o = init_weights()
    LSTMScratch(
        (
            input_gate = (W_ix = W_ix, W_ih = W_ih, b_i = b_i),
            forget_gate = (W_fx = W_fx, W_fh = W_fh, b_f = b_f),
            input_node = (W_cx = W_cx, W_ch = W_ch, b_c = b_c),
            output_gate = (W_ox = W_ox, W_oh = W_oh, b_o = b_o)
        ),
        (num_inputs = num_inputs, num_hiddens = num_hiddens, sigma = sigma)
    )
            
        
    
    
end

function (model::LSTMScratch)(x, state = nothing)
    weights = model.weights
    device = isa(x, CuArray) ? gpu : cpu 
    batchsize = size(x, 3)
    H, C = if isnothing(state)
        zeros(model.args.num_hiddens, batchsize), zeros(model.args.num_hiddens, batchsize)
    else
        state 
    end |> device

    output = map(eachslice(x; dims= 2)) do x_ 
        It = sigmoid.(weights.input_gate.W_ix*x_ + weights.input_gate.W_ih*H .+ weights.input_gate.b_i)
        Ft = sigmoid.(weights.forget_gate.W_fx*x_ + weights.forget_gate.W_fh*H .+ weights.forget_gate.b_f)
        Ot = sigmoid.(weights.output_gate.W_ox*x_ + weights.output_gate.W_oh*H .+ weights.output_gate.b_o)
        C_tilde = tanh.(weights.input_node.W_cx*x_ + weights.input_node.W_ch*H .+ weights.input_node.b_c) # candidate cell state
        C = Ft.*C + It.*C_tilde
        H = Ot.*C
        H
    end
    out = stack(output)
    return permutedims(out, [1, 3, 2]), (H,C)
end


struct GRUScratch{W, A} <: AbstractModel 
    w::W 
    args::A
end

Flux.@layer GRUScratch trainable = (w,)
function GRUScratch(num_inputs::Int, num_hiddens::Int; sigma = 0.01)
    init_weights() = randn(num_hiddens, num_inputs).*sigma, randn(num_hiddens, num_hiddens).*sigma, zeros(num_hiddens)

    W_rx, W_rh, b_r = init_weights()
    W_zx, W_zh, b_z = init_weights()
    W_cx, W_ch, b_c = init_weights()

    w = (
        reset_gate = construct_nt_args(; W_rx, W_rh, b_r),
        update_gate = construct_nt_args(; W_zx, W_zh, b_z),
        input_node = construct_nt_args(; W_cx, W_ch, b_c)
    )

    args = construct_nt_args(; num_inputs, num_hiddens, sigma)

    GRUScratch(w, args)
end

function (gru::GRUScratch)(x, state = nothing)
    batchsize = size(x, 3)
    device = isa(x, CuArray) ? gpu : cpu
    H = if isnothing(state) 
        zeros(gru.args.num_hiddens, batchsize)
    else
        state 
    end |> device

    outs = map(eachslice(x; dims = 2)) do x_ 
        Rt = sigmoid.(gru.w.reset_gate.W_rx*x_ + gru.w.reset_gate.W_rh*H .+ gru.w.reset_gate.b_r)
        Zt = sigmoid.(gru.w.update_gate.W_zx*x_ + gru.w.update_gate.W_zh*H .+ gru.w.update_gate.b_z)
        H_tilde = tanh.(gru.w.input_node.W_cx*x_ + gru.w.input_node.W_ch*(H.*Rt) .+ gru.w.input_node.b_c)
        H = Zt.*H + (1. .- Zt).*H_tilde
        return H 
    end
    outputs = stack(outs)
    permutedims(outputs, [1,3,2]), H
end

struct StackedRNN{N, A} <: AbstractModel 
    net::N 
    args::A 
end 
Flux.@layer StackedRNN trainable = (net,)
function StackedRNN(num_inputs, num_hiddens, num_layers; rnn = GRU, init = Flux.glorot_uniform)
    layers = map(1:num_layers) do i
        if i==1 
            return GRU(num_inputs => num_hiddens; return_state = true, init_kernel = init, init_recurrent_kernel = init)
        else
            return GRU(num_hiddens => num_hiddens; return_state = true, init_kernel = init, init_recurrent_kernel = init)
        end
    end
    StackedRNN(layers, construct_nt_args(; num_inputs, num_hiddens, num_layers))
end

# function (m::StackedRNN)(x, state = nothing)
#     states0 = isnothing(state) ? [Flux.initialstates(n) for n in m.net] : state 
#     states = map(m.net, states0) do layer, state0 
#         x, state_ = layer(x, state0)
#         return state_
#     end
#     x, states

# end
function (m::d2lai.StackedRNN)(x, state = nothing)
    states0 = isnothing(state) ? [Flux.initialstates(n) for n in m.net] : state
    x_out = x  # Keep track of transformed input
    states_out = []
    for (i, layer) in enumerate(m.net)
        x_out, new_state = layer(x_out, states0[i])
        states_out = [states_out; [new_state]]
    end
    
    return x_out, states_out
end


abstract type AbstractEncoderDecoder <: AbstractClassifier end 

function init_state end

function (model::AbstractEncoderDecoder)(enc_X, dec_X, args)
    enc_all_outputs = model.encoder(enc_X, args)
    dec_state = init_state(model.decoder, enc_all_outputs, args)
    return model.decoder(dec_X, dec_state)[1]
end


struct Seq2SeqEncoder{E, R , A} <: AbstractModel 
    embedding::E
    rnn::R
    args::A
end

function Seq2SeqEncoder(vocab_size, embed_size, num_hiddens, num_layers, dropout=0)
    embedding = Embedding(vocab_size => embed_size)
    rnn = StackedRNN(embed_size, num_hiddens, num_layers)
    args = (; vocab_size, embed_size, num_hiddens, num_layers)
    Seq2SeqEncoder(embedding, rnn, args)
end

function (m::Seq2SeqEncoder)(x, args)
    embs = m.embedding(x)
    out, state = m.rnn(embs)
    return out, state
end


struct Seq2Seq{E, D, T, A} <: d2lai.AbstractEncoderDecoder
    encoder::E 
    decoder::D 
    tgt_pad::T 
    args::A 
end 

function Seq2Seq(encoder::AbstractModel, decoder::AbstractModel, tgt_pad)
    return Seq2Seq(encoder, decoder, tgt_pad, (;))
end


function d2lai.loss(model::AbstractEncoderDecoder, y_pred, y)
    # Compute per-token cross entropy loss (shape: vocab × seq_len × batch)
    target_oh = Flux.onehotbatch(y, 1:model.decoder.args.vocab_size)
    loss = Flux.logitcrossentropy(y_pred, target_oh; agg = Flux.identity)

    # Create mask (ensure it's same type and device as loss)
    mask = reshape(y, 1, 9, :) .!= model.tgt_pad
    mask = eltype(loss).(mask)

    # Apply mask and normalize
    masked_loss = mask .* loss
    return sum(masked_loss) / (sum(mask) + eps(eltype(masked_loss)))  # to avoid divide-by-zero
end

function d2lai.training_step(m::AbstractEncoderDecoder, batch)
    y_pred = d2lai.forward(m, batch[1:end-1]...)
    loss_ = d2lai.loss(m, y_pred, batch[end])
    return loss_
end

function d2lai.validation_step(m::AbstractEncoderDecoder, batch)
    y_pred = d2lai.forward(m, batch[1:end-1]...)
    loss_ = d2lai.loss(m, y_pred, batch[end])
    return loss_ , nothing
end