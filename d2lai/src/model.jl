abstract type AbstractModel end
abstract type AbstractClassifier <: AbstractModel end
abstract type AbstractConciseModel <: AbstractModel end

struct Model{N, A} <: AbstractModel
    net::N
    args::A
end

function loss(model::AbstractModel, y_pred, y)
end

function loss(model::AbstractClassifier, y_pred, y)
    Flux.crossentropy(y_pred, Flux.onehotbatch(y, 0:9))
end

function forward(model::AbstractModel, x)
    model(x)
end

function accuracy(model::AbstractClassifier, y_pred, y; averaged = true)
    y_labels = argmax(y_pred, dims = 1)
    y_labels = getindex.(y_labels , 1)
    compare = (y_labels .== (y' .+ 1))
    return averaged ? mean(compare) : compare
end

# function accuracy(model::AbstractClassifier, y_pred, y; averaged = true)
#     y_labels = argmax(y_pred, dims = 1)
#     y_labels = getindex.(y_labels , 1)
#     compare = (y_labels .== (y' .+ 1))
#     return averaged ? mean(compare) : compare
# end

function training_step(m::AbstractModel, batch)
    y_pred = forward(m, batch[1])
    loss_ = loss(m, y_pred, batch[end])
    return loss_
end

function validation_step(m::AbstractModel, batch)
    y_pred = forward(m, batch[1])
    loss_ = loss(m, y_pred, batch[end])
    return loss_ , nothing
end

function validation_step(m::AbstractClassifier, batch)
    y_pred = forward(m, batch[1])
    loss_ = loss(m, y_pred, batch[end])
    acc = accuracy(m, y_pred, batch[end])
    return loss_, acc
end

function configure_optimizers(m::AbstractModel)
end


abstract type AbstractRNNClassifier <: AbstractClassifier end


function d2lai.loss(m::AbstractRNNClassifier, y_pred, y)
    Flux.logitcrossentropy(y_pred, Flux.onehotbatch(y, 1:m.args.vocab_size))
end

function d2lai.training_step(m::AbstractRNNClassifier, batch)
    y_pred = d2lai.forward(m, batch[1])
    loss_ = d2lai.loss(m, y_pred, batch[end])
    return loss_
end

function d2lai.validation_step(m::AbstractRNNClassifier, batch)
    y_pred = d2lai.forward(m, batch[1])
    loss_ = d2lai.loss(m, y_pred, batch[end])
    return loss_ , nothing
end

function output_layer end

function prediction(prefix, model::AbstractRNNClassifier, vocab, num_preds)
    outputs = [vocab.token_to_idx[string(prefix[1])]]
    state = zeros(32)
    for i in 2:length(prefix)
        x = outputs[end]
        x = reshape(Flux.onehotbatch(x, 1:length(vocab)), :, 1, 1)
        _, state = model.rnn(x, state)
        push!(outputs, vocab.token_to_idx[string(prefix[i])])
    end
    for i in 1:num_preds 
        x = outputs[end]
        x = reshape(Flux.onehotbatch(x, 1:length(vocab)), :, 1, 1)
        out, state = model.rnn(x, state)
        out = output_layer(model, out)
        idx = argmax(softmax(out), dims = 1)[1][1]
        push!(outputs, idx)
    end
    out_chars = map(outputs) do o 
        vocab.idx_to_token[o]
    end
    join(out_chars)
end

