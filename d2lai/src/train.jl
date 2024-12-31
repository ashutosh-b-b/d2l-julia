abstract type AbstractTrainer end 

struct Trainer{M, D, P, O, A} <: AbstractTrainer
    model::M 
    data::D 
    board::P
    opt::O
    args::A
    function Trainer(model::AbstractModel, data, opt; board_yscale = :log10, args...)
        board = ProgressBoard("epochs", ["train_loss" "val_loss"]; yscale = board_yscale)
        default_args = (verbose = true,)
        args = merge(default_args, args)
        new{typeof(model), typeof(data), typeof(board), typeof(opt), NamedTuple}(
            model, data, board, opt, NamedTuple(args)
        )
    end
    function Trainer(model::AbstractClassifier, data, opt; board_yscale = :log10, args...)
        board = ProgressBoard("epochs", ["train_loss" "val_loss" "val_acc"]; yscale = board_yscale)
        default_args = (verbose = true,)
        args = merge(default_args, args)
        new{typeof(model), typeof(data), typeof(board), typeof(opt), NamedTuple}(
            model, data, board, opt, NamedTuple(args)
        )
    end
end
function prepare_data(data::AbstractData, train_on_gpu = false)
    train_dataloader = d2lai.train_dataloader(data) 
    val_dataloader = d2lai.val_dataloader(data)
    train_dataloader = train_on_gpu ? gpu(train_dataloader) : train_dataloader
    val_dataloader = train_on_gpu ? gpu(val_dataloader) : val_dataloader
    return train_dataloader, val_dataloader
end

function prepare_model(model::AbstractModel, train_on_gpu)
    return train_on_gpu ? gpu(model) : model
end

function draw(trainer::AbstractTrainer, epoch, losses, n_batches, label)
    x = epoch .+ (collect(1:n_batches)/n_batches)
    draw.(Ref(trainer.board), x, losses, Ref(label))
    
end
function fit(trainer::Trainer)
    train_on_gpu = get(trainer.args, :gpu, false)
    td, vd = prepare_data(trainer.data, train_on_gpu)
    model = prepare_model(trainer.model, train_on_gpu)
    num_train_batches = length(td)
    num_val_batches = length(vd)
    for epoch in 1:trainer.args.max_epochs 
        losses = fit_epoch(model, trainer.opt; train_dataloader = td, val_dataloader = vd)
        draw(trainer, epoch, losses.train_losses, num_train_batches, "train_loss")
        draw(trainer, epoch, losses.val_losses, num_val_batches, "val_loss")
        trainer.args.verbose && @info "Train Loss: $(losses.train_losses[end]), Val Loss: $(losses.val_losses[end]), Val Acc: $(losses.val_acc[end])"
        !isempty(losses.val_acc) && draw(trainer, epoch, losses.val_acc, num_val_batches, "val_acc")
    end
    trainer.args.verbose && display(trainer.board.plt)
    final_val_metrics = validation_step.(Ref(model), vd)
    final_val_loss = mean.(getindex.(final_val_metrics,1))
    final_val_acc = mean.(getindex.(final_val_metrics, 2))
    return cpu(model) , (val_loss = final_val_loss, val_acc = final_val_acc)
end

function fit_epoch(trainer::AbstractTrainer) end