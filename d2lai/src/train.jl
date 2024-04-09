abstract type AbstractTrainer end 

struct Trainer{M, D, P, O, A} <: AbstractTrainer
    model::M 
    data::D 
    board::P
    opt::O
    args::A
    function Trainer(model::AbstractModel, data, opt; args...)
        board = ProgressBoard("epochs", ["train_loss", "val_loss"]; yscale = :log10)
        new{typeof(model), typeof(data), typeof(board), typeof(opt), NamedTuple}(
            model, data, board, opt, NamedTuple(args)
        )
    end
    function Trainer(model::AbstractClassifier, data, opt; args...)
        board = ProgressBoard("epochs", ["train_loss", "val_loss", "val_acc"]; yscale = :log10)
        new{typeof(model), typeof(data), typeof(board), typeof(opt), NamedTuple}(
            model, data, board, opt, NamedTuple(args)
        )
    end
end
function prepare_data(data::AbstractData)
    train_dataloader = d2lai.train_dataloader(data)
    val_dataloader = d2lai.val_dataloader(data)
    return train_dataloader, val_dataloader
end
function prepare_model(model::AbstractModel)
end

function draw(trainer::AbstractTrainer, epoch, losses, n_batches, label)
    x = epoch .+ (collect(1:n_batches)/n_batches)
    draw.(Ref(trainer.board), x, losses, Ref(label))
    
end
function fit(trainer::Trainer)
    td, vd = prepare_data(trainer.data)
    prepare_model(trainer.model)
    num_train_batches = length(td)
    num_val_batches = length(vd)
    for epoch in 1:trainer.args.max_epochs 
        losses = fit_epoch(trainer; train_dataloader = td, val_dataloader = vd)
        draw(trainer, epoch, losses.train_losses, num_train_batches, "train_loss")
        draw(trainer, epoch, losses.val_losses, num_val_batches, "val_loss")
        !isempty(losses.val_acc) && draw(trainer, epoch, losses.val_acc, num_val_batches, "val_acc")
    end
end

function fit_epoch(trainer::AbstractTrainer) end