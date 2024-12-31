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
    model.net(x)
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
    return loss_ 
end

function validation_step(m::AbstractClassifier, batch)
    y_pred = forward(m, batch[1])
    loss_ = loss(m, y_pred, batch[end])
    acc = accuracy(m, y_pred, batch[end])
    return loss_, acc
end

function configure_optimizers(m::AbstractModel)
end

