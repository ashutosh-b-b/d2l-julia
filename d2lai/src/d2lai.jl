module d2lai

using Flux, Plots
using Random, Distributions
include("data.jl")

export SyntheticRegressionData, get_dataloader, train_dataloader, val_dataloader, AbstractData

include("model.jl")
export AbstractModel, AbstractClassifier, forward, training_step, validation_step, configure_optimizers

include("plotting.jl")
export ProgressBoard, draw, save_gif, save_plot

include("train.jl")
export AbstractTrainer, Trainer, draw, fit
end
