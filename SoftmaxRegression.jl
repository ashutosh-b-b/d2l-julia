
function softmax(o)
    return exp.(o) ./ sum(exp.(o), dims = 1)
end

struct SoftmaxRegressionScratch{A} <: AbstractClassifier 
    w::AbstractArray
    b::AbstractArray
    args::A
    function SoftmaxRegressionScratch(num_inputs, num_outputs, lr, sigma=0.01)
        w = rand(Normal(0, sigma), (num_outputs, num_inputs))
        b = zeros(num_outputs, 1)
        args = (num_inputs = num_inputs, num_outputs = num_outputs, lr = lr, sigma = sigma)
        new{typeof(args)}(w, b, args)
    end
end

function d2lai.forward(model::SoftmaxRegressionScratch, x)
    softmax(model.w * x  .+ model.b)
end

function d2lai.loss(model::AbstractClassifier, y_pred, y)
    # cross entropy 
    actual_class_prob = getindex.(eachcol(y_pred), y .+ 1)
    return mean(-1*log.(actual_class_prob))
end

function d2lai.accuracy(model::AbstractClassifier, y_pred, y; averaged = true)
    y_labels = argmax(y_pred, dims = 1)
    y_labels = getindex.(y_labels , 1)
    compare = (y_labels .== (y' .+ 1))
    return averaged ? mean(compare) : compare
end

function d2lai.fit_epoch(trainer::Trainer{<:SoftmaxRegressionScratch}; train_dataloader = nothing, val_dataloader = nothing)
    losses = (train_losses = [], val_losses = [], val_acc = [])
    for batch in train_dataloader
        ps = Flux.Params([model.w, model.b])
        gs = gradient(ps) do 
            training_step(trainer.model, batch)
        end
        Flux.Optimise.update!(trainer.opt, ps, gs)
        train_loss = training_step(trainer.model, batch)
        push!(losses.train_losses, train_loss)
    end
    for batch in val_dataloader
        loss, acc = validation_step(trainer.model, batch)
        push!(losses.val_losses , loss)
        push!(losses.val_acc, acc)
    end
    return losses
end
model = SoftmaxRegressionScratch(784, 10, 0.1)
opt = Descent(0.01)
data = FashionMNISTData(; batchsize = 256)
trainer = Trainer(model, data, opt; max_epochs = 10)
d2lai.fit(trainer)

#================================ SOFTMAX REGRESSION CONCISE =====================================# 

struct SoftmaxRegressionConcise{A,N} <: AbstractClassifier 
    args::A
    net::N
    function SoftmaxRegressionConcise(net; args...)
        new{typeof(args), typeof(net)}(args, net)
    end
end

d2lai.forward(model::SoftmaxRegressionConcise, x) = model.net(x)

function d2lai.loss(model::SoftmaxRegressionConcise, y_pred, y)
    return Flux.crossentropy(y_pred, Flux.onehotbatch(y, 0:9))
end

function d2lai.fit_epoch(trainer::Trainer{<:SoftmaxRegressionConcise}; train_dataloader = nothing, val_dataloader = nothing)
    losses = (train_losses = [], val_losses = [], val_acc = [])
    for batch in train_dataloader
        ps = Flux.params(trainer.model.net)
        gs = gradient(ps) do 
            training_step(trainer.model, batch)
        end
        Flux.Optimise.update!(trainer.opt, ps, gs)
        train_loss = training_step(trainer.model, batch)
        push!(losses.train_losses, train_loss)
    end
    for batch in val_dataloader
        loss, acc = validation_step(trainer.model, batch)
        push!(losses.val_losses , loss)
        push!(losses.val_acc, acc)
    end
    return losses
end

net = Chain(Dense(28*28 => 10), Flux.softmax)
model = SoftmaxRegressionConcise(net; num_inputs = 784, num_outputs = 10, lr = 0.01)
opt = Descent(0.01)
data = FashionMNISTData(; batchsize = 256)
trainer = Trainer(model, data, opt; max_epochs = 10)
d2lai.fit(trainer)
