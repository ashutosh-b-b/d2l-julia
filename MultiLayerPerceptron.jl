#================================ MLP =====================================# 

struct MLP <: AbstractClassifier 
    W1::AbstractArray
    W2::AbstractArray
    B1::AbstractArray 
    B2::AbstractArray 
    args::NamedTuple
    function MLP(num_inputs, num_outputs, num_hiddens, lr, sigma = 0.01)
        W1 = rand(Normal(0, sigma), (num_hiddens, num_inputs))
        B1 = zeros(num_hiddens, 1)
        W2 = rand(Normal(0, sigma), (num_outputs, num_hiddens))
        B2 = zeros(num_outputs, 1)
        args = (num_inputs = num_inputs, num_hiddens = num_hiddens, num_outputs = num_outputs, lr = lr)
        new(W1, W2, B1, B2, args)
    end
end

function d2lai.forward(m::MLP, x)
    # @info size(x)
    # @info size(model.W1)
    # @info size(model.B1)
    H = relu_custom.(m.W1*x .+ m.B1)
    O = softmax(m.W2*H .+ m.B2)
    return O
end

function d2lai.loss(m::MLP, y_pred, y)
    # cross entropy 
    # y_pred is an array of n_outputs x batchsize 
    # y actual is a vector of labels 
    y_prob = getindex.(eachcol(y_pred), y .+ 1)
    mean(-1*log.(y_prob))
end

function softmax(o::AbstractArray)
    # o = num_outputs x batchsize
    exp.(o) ./ sum(exp.(o), dims = 1)
end

function relu_custom(x)
    return max(x, 0.0)
end

function d2lai.fit_epoch(trainer::Trainer{<:MLP}; train_dataloader = nothing, val_dataloader = nothing)
    losses = (train_losses = [], val_losses = [], val_acc = [])
    for batch in train_dataloader
        ps = Flux.Params([model.W1, model.W2, model.B1, model.B2])
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

model = MLP(28*28, 10, 256, 0.01)
opt = Descent(0.01)
data = FashionMNISTData(; batchsize = 256)
trainer = Trainer(model, data, opt; max_epochs = 10)
d2lai.fit(trainer)

#================================ MLP CONCISE =====================================# 

struct MLPConcise{N, A} <: AbstractClassifier 
    net::N 
    args::A
    function MLPConcise(num_inputs, num_outputs, num_hiddens, lr, sigma = 0.01)
        args = (num_inputs = num_inputs, num_hiddens = num_hiddens, num_outputs = num_outputs, lr = lr)
        net = Chain(Dense(num_inputs, num_hiddens, relu), Dense(num_hiddens, num_outputs), Flux.softmax)
        new{typeof(net), typeof(args)}(net, args)
    end
end

d2lai.forward(m::MLPConcise, x) = m.net(x)
d2lai.loss(m::MLPConcise, y_pred, y) = Flux.crossentropy(y_pred, Flux.onehotbatch(y, 0:9))

function d2lai.fit_epoch(trainer::Trainer{<:Union{MLPConcise, DropoutScratchMLP}}; train_dataloader = nothing, val_dataloader = nothing)
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

model = MLPConcise(28*28, 10, 256, 0.01)
opt = Descent(0.01)
data = FashionMNISTData(; batchsize = 256)
trainer = Trainer(model, data, opt; max_epochs = 10)
d2lai.fit(trainer)

#================================ DropoutScratchMLP =====================================# 


mutable struct DropoutScratchMLP{N, A} <: AbstractClassifier
    net::N 
    args::A 
    train::Bool
    function DropoutScratchMLP(; args...)
        net = Chain(Dense(args[:num_inputs], args[:num_hidden_1]), Dense(args[:num_hidden_1], args[:num_hidden_2]), Dense(args[:num_hidden_2], args[:num_outputs]), Flux.softmax)
        new{typeof(net), NamedTuple}(net, NamedTuple(args), true)
    end

end
function dropout_layer(X::AbstractArray, dropout)
    probs = rand(Bernoulli(dropout), size(X, 1))
    return probs .* X
end

function d2lai.forward(mlp::DropoutScratchMLP, x)
    lin1, lin2, lin3, softmax = mlp.net.layers
    h1 = model.train ? dropout_layer(lin1(x), mlp.args.dropout_1) : lin1(x)
    h2 = model.train ? dropout_layer(lin2(h1), mlp.args.dropout_2) : lin2(h1)
    h3 = lin3(h2)
    return softmax(h3)
end

function d2lai.loss(mlp::AbstractClassifier, y_pred, y)
    Flux.crossentropy(y_pred, Flux.onehotbatch(y, 0:9))
end

hparams = (num_inputs = 28*28, num_outputs = 10, num_hidden_1 = 256, num_hidden_2 = 256,
           dropout_1 = 0.5, dropout_2 = 0.5, lr = 0.1)
model = DropoutScratchMLP(; hparams...)

opt = Descent(0.1)
data = FashionMNISTData(; batchsize = 256)
trainer = Trainer(model, data, opt; max_epochs = 10)
d2lai.fit(trainer)

