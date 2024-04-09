#= Notes 
- Assumes relation between features `x` and target `y` is linear.
- E[Y | X = x] can be represented as linear combination of `x`
=#
using d2lai.Flux 
struct LinearRegressionScratch <: AbstractModel
    w::AbstractArray 
    b::AbstractArray 
    args::NamedTuple
    function LinearRegressionScratch(num_inputs, lr, sigma = 0.01)
        w = rand(Normal(0, sigma), 1, num_inputs)
        b = zeros(1)
        args = (num_inputs = num_inputs, lr = lr, sigma = sigma)
        new(w, b, args)
    end
end

function d2lai.forward(lr::LinearRegressionScratch, x)
    return lr.w*x .+ lr.b
end

function d2lai.loss(lr::LinearRegressionScratch, y_pred, y)
    l = 0.5*(y_pred - y).^2
    return mean(l)
end

struct SGD{P, L}
    params::P
    lr::L
end

function step!(sgd::SGD, model, grads)
    model.w .-= sgd.lr.*grads[model.w]
    model.b .-= sgd.lr.*grads[model.b]
end
step!(tr::Trainer, args...) = step!(tr.opt, tr.model, args...)

function d2lai.fit_epoch(trainer::Trainer; train_dataloader = nothing , val_dataloader = nothing)
    losses = (train_losses = [], val_losses = [])
    for batch in train_dataloader
        gs = gradient(Flux.Params([model.w, model.b])) do 
            training_step(trainer.model, batch)
        end 
        step!(trainer, gs)
        train_loss = training_step(trainer.model, batch)
        push!(losses.train_losses, train_loss)
    end
    for batch in val_dataloader
        loss = validation_step(trainer.model, batch)
        push!(losses.val_losses , loss)
    end
    return losses
end
model = LinearRegressionScratch(2, 0.03)
sgd = SGD(nothing, 0.03)
data = SyntheticRegressionData([2 -3.4], 4.3)

trainer = Trainer(model, data, sgd; max_epochs = 5)

d2lai.fit(trainer)

trainer.board.plt 

@show model.w
@show model.b 

#================================ LINEAR REGRESSION CONCISE =====================================# 
struct LinearRegressionConcise{N} <: AbstractModel 
    net::N
end

function d2lai.forward(lr::LinearRegressionConcise, x)
    lr.net(x)
end

function d2lai.loss(lr::LinearRegressionConcise, y_pred, y)
    Flux.Losses.mse(y_pred, y)
end

opt = Descent(0.03)

function d2lai.fit_epoch(trainer::Trainer{<:LinearRegressionConcise}; train_dataloader = nothing, val_dataloader = nothing)
    losses = (train_losses = [], val_losses = [])
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
        loss = validation_step(trainer.model, batch)
        push!(losses.val_losses , loss)
    end
    return losses
end

model = LinearRegressionConcise(Dense(2 => 1))
opt = Descent(0.03)
data = SyntheticRegressionData([2 -3.4], 4.3)
trainer = Trainer(model, data, opt; max_epochs = 3)
d2lai.fit(trainer)

#================================ WEIGHT DECAY =====================================# 
struct PolynomialData{XT, YT, A} <: AbstractData 
    X::XT 
    y::YT
    args::A 
    function PolynomialData(num_train, num_val, num_inputs, batch_size)
        args = (num_train = num_train, num_val = num_val, num_inputs = num_inputs, batchsize = batch_size)
        n = num_train + num_val 
        X = randn(num_inputs, n)
        b = zeros(1)
        y = 0.01*ones(1, num_inputs)*X .+ b .+ 0.01*randn(1, n)
        new{typeof(X), typeof(y), typeof(args)}(X, y, args)
    end
end
function get_dataloader(data::PolynomialData; train = true)
    if train 
        return Flux.DataLoader((data.X[:, 1:data.args.num_train], data.y[:, 1:data.args.num_train]), batchsize = data.args.batchsize, shuffle=true)
    else
        return Flux.DataLoader((data.X[:, data.args.num_train + 1 : end], data.y[:, data.args.num_train + 1 : end]), batchsize = data.args.batchsize)
    end
end

struct WeightDecayScratch{N, A} <: AbstractModel 
    net::N
    args::A
    function WeightDecayScratch(net, lambda = 0.01)
        args = (lambda = lambda, )
        return new{typeof(net), typeof(args)}(net, args)
    end
end

d2lai.forward(m::WeightDecayScratch, x) = m.net(x)

function d2lai.loss(m::WeightDecayScratch, y_pred, y)
    mse_loss = Flux.Losses.mse(y_pred, y)
    reg_loss = m.args.lambda*sum(m.net.weight.^2)
    return mse_loss + reg_loss
end

function d2lai.fit_epoch(trainer::Trainer{<:WeightDecayScratch}; train_dataloader = nothing, val_dataloader = nothing)
    losses = (train_losses = [], val_losses = [])
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
        loss = validation_step(trainer.model, batch)
        push!(losses.val_losses , loss)
    end
    return losses
end

model = WeightDecayScratch(Dense(200 => 1), 0.0)
opt = Descent(0.01)
data = PolynomialData(20, 100, 200, 5)
trainer = Trainer(model, data, opt; max_epochs = 10)
d2lai.fit(trainer)

model = WeightDecayScratch(Dense(200 => 1), 3.0)
opt = Descent(0.01)
data = PolynomialData(20, 100, 200, 5)
trainer = Trainer(model, data, opt; max_epochs = 10)
d2lai.fit(trainer)