module d2lai

using Flux, Plots
using Random, Distributions
using MLDatasets
using Images
const DATA_URL = "http://d2l-data.s3-accelerate.amazonaws.com/"
include("data.jl")

export SyntheticRegressionData, get_dataloader, train_dataloader, val_dataloader, AbstractData

include("misc.jl")

include("model.jl")
export AbstractModel, AbstractClassifier, forward, training_step, validation_step, configure_optimizers

include("plotting.jl")
export ProgressBoard, draw, save_gif, save_plot

include("train.jl")
export AbstractTrainer, Trainer, draw, fit

include("fit.jl")

include("models/linear_regression.jl")
export LinearRegressionConcise
end
