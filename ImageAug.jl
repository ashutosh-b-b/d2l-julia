# using DataAugmentation, Images
# using d2lai, Flux, MLDatasets

# function apply_(img, aug; nrows = 2, ncols = 4, scale = 1.5)
#     item = Image(img)
#     Y = map(1:(nrows*ncols)) do _ 
#         apply(aug, item) |> itemdata
#     end
#     mosaicview(Y...; nrow = nrows)
# end

# img = load("./Julia_Notebooks/img/cat1.jpg")

# aug = DataAugmentation.compose(Maybe(FlipX{2}()), ImageToTensor())
#     item = Image(img)
# apply(aug, item)  |> itemdata |> collect
# apply_(img, aug)


# aug = Maybe(FlipY{2}())
# apply_(img, aug)


# aug = RandomCrop((200,200))
# apply_(img, aug)

# aug = DataAugmentation.compose(AdjustBrightness(0.5), AdjustContrast(0.5))
# apply_(img, aug)

# dataset = MLDatasets.CIFAR10
# d = dataset(:train)[:]
# d.features


using d2lai
using DataAugmentation, Images
using Flux, MPI, NCCL, CUDA
using Random
using Flux.Optimisers
using Flux.Zygote
using Statistics
using MLDatasets
using Plots

struct CIFAR10Data{T,V,L,A} <: d2lai.AbstractData 
    train::T
    val::V
    labels::L
    args::A
    function CIFAR10Data(; batchsize = 64, flatten = false, aug = nothing)
        dataset = MLDatasets.CIFAR10
        t = dataset(:train)[:]
        v = dataset(:test)[:]
       
        l = dataset().metadata["class_names"]
        train, val = map((t,v)) do x
            item = colorview(RGB, permutedims(x.features[:,:,:,:], (3, 1, 2, 4)))
            item = Image(item)
            data = itemdata(apply(aug, item))
            data = permutedims(data, (1, 2, 4, 3)) |> collect
            (features = data, targets= x.targets)
        end
        args = (batchsize = batchsize, flatten = flatten)
        new{typeof(train), typeof(val), typeof(l), typeof(args)}(train, val, l, args)
    end
end


aug = DataAugmentation.compose(
    Maybe(FlipX{3}()),
    ImageToTensor()
)
data = CIFAR10Data(; aug)

arch = ((2, 64), (2, 128), (2, 256), (2, 512))
model = ResNet(arch, 10, (32, 32, 3)) |> gpu ;


# using Flux, MPI, NCCL, CUDA
# using Random
# using Flux.Optimisers
# using Flux.Zygote
# using Statistics
# using d2lai 

CUDA.allowscalar(false)

# DistributedUtils.initialize(NCCLBackend)
# backend = DistributedUtils.get_distributed_backend(NCCLBackend)
# rank = DistributedUtils.local_rank(backend)

# model = Chain(Dense(1 => 256, tanh), Dense(256, 256, tanh), Dense(256 => 1)) |> gpu

# struct ModelP{N} <: AbstractModel 
#   net::N
# end

# Flux.@layer ModelP 

# model = ModelP(model)

# (m::ModelP)(x) = m.net(x)
# model = DistributedUtils.synchronize!!(backend, DistributedUtils.FluxDistributedModel(model); root=0) 

# x = rand(Float32, 1, 1028) |> gpu
# y = x .^ 3

# # data = DistributedUtils.DistributedDataContainer(
# #             backend, (x, y)
# #         )

# # train_loader = Flux.DataLoader(data, batchsize = 64)
# # opt = DistributedUtils.DistributedOptimizer(backend, Optimisers.Adam(0.001f0))
# # st_opt = Optimisers.setup(opt, model)
# # st_opt = DistributedUtils.synchronize!!(backend, st_opt; root=0) 

# loss(model, x, y) = mean((model(x) .- y).^2)
# g_ = Zygote.gradient(m -> loss(m), model)[1] 
# Optimisers.update!(st_opt, model, g_)

# function train_model(model, st_opt)
#   for epoch in 1:100
#     for d in train_loader
#       l, back = Zygote.pullback(loss, model, d[1], d[2])
#       println("Epoch $epoch: Loss $l")
#       g = back(one(l))[1]
#       st_opt, model = Optimisers.update(st_opt, model, g)
#     end
#   end
# end

# train_model(model, st_opt)

function train_ch13(model, train_iter, test_iter, trainer;  num_epochs = 100, batchsize = 256, verbose = true)
  DistributedUtils.initialize(NCCLBackend)
  backend = DistributedUtils.get_distributed_backend(NCCLBackend)
  rank = DistributedUtils.local_rank(backend)
  model = DistributedUtils.synchronize!!(backend, DistributedUtils.FluxDistributedModel(model); root=0) 
  opt = DistributedUtils.DistributedOptimizer(backend, trainer.opt)
  st_opt = Optimisers.setup(opt, model)
  st_opt = DistributedUtils.synchronize!!(backend, st_opt; root=0) 
  train_data = DistributedUtils.DistributedDataContainer(
            backend, train_iter
        )

  train_loader = Flux.DataLoader(train_data, batchsize = batchsize, shuffle = true)

  val_data = DistributedUtils.DistributedDataContainer(
            backend, test_iter
        )

  val_loader = Flux.DataLoader(val_data, batchsize = batchsize)

  for i in 1:num_epochs 
    losses = (train_losses = [], val_losses = [], val_acc = [])
    for batch in train_loader 
      l, back = Zygote.pullback(d2lai.training_step, model, batch)
      g = back(one(l))[1]
      st_opt, model = Optimisers.update(st_opt, model, g)
      push!(losses.train_losses, d2lai.training_step(model, batch))
    end
    for batch in val_loader 
      val_loss, val_acc = d2lai.validation_step(model, batch)
      push!(losses.val_losses, val_loss)
      push!(losses.val_acc, val_acc)
    end
    verbose &&@info "Epoch: $i Training Loss: $(mean(losses.train_losses)) Val Loss: $(mean(losses.val_losses)) Val Acc: $(mean(losses.val_acc))" 

    d2lai.draw_metrics(model, i, trainer, losses)
  end
  verbose && Plots.display(trainer.board.plt)
end

trainer = Trainer(model, nothing, Optimisers.Adam(0.01))


# train_ch13(model, (x,y), nothing, loss, Optimisers.Adam(0.01), 100)

function load_cifar10(aug, is_train)
  data = CIFAR10Data(; aug)
  is_train ? data.train : data.val 
end


train_iter = load_cifar10(aug, true) |> gpu
test_iter = load_cifar10(aug, false) |> gpu

train_ch13(model, train_iter, test_iter, trainer; num_epochs= 20)

