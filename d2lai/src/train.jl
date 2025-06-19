abstract type AbstractTrainer end 

struct Trainer{M, D, P, O, A} <: AbstractTrainer
    model::M 
    data::D 
    board::P
    opt::O
    args::A
    function Trainer(model::AbstractModel, data, opt; board_yscale = :log10, metrics = ["train_loss" "val_loss"], args...)
        board = ProgressBoard("epochs", []; yscale = board_yscale)
        default_args = (verbose = true, gradient_clip_val = 0.)
        args = merge(default_args, args)
        new{typeof(model), typeof(data), typeof(board), typeof(opt), NamedTuple}(
            model, data, board, opt, NamedTuple(args)
        )
    end
    function Trainer(model::AbstractClassifier, data, opt; board_yscale = :log10, args...)
        board = ProgressBoard("epochs", []; yscale = board_yscale)
        default_args = (verbose = true, gradient_clip_val = 0.)
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

function draw_metrics(model::AbstractModel, epoch, trainer::Trainer, metrics)
    draw(trainer.board, epoch, mean(metrics.train_losses), "train_loss")
    draw(trainer.board, epoch, mean(metrics.val_losses), "val_loss")
end

function draw_metrics(model::AbstractClassifier, epoch, trainer::Trainer, metrics)
    draw(trainer.board, epoch, mean(metrics.train_losses), "train_loss")
    draw(trainer.board, epoch, mean(metrics.val_losses), "val_loss")
    !isempty(metrics.val_acc) && draw(trainer.board, epoch, mean(metrics.val_acc), "val_acc")
end

function draw_metrics(model::AbstractRNNClassifier, epoch, trainer::Trainer, metrics)
    draw(trainer.board, epoch, exp(mean(metrics.train_losses)), "train_ppl")
    draw(trainer.board, epoch, exp(mean(metrics.val_losses)), "val_ppl")
    !isempty(metrics.val_acc) && draw(trainer.board, epoch, mean(metrics.val_acc), "val_acc")
end

function fit(trainer::Trainer)
    train_on_gpu = get(trainer.args, :gpu, false)
    print_every = get(trainer.args, :print_every, 1)
    td, vd = prepare_data(trainer.data, train_on_gpu)
    model = prepare_model(trainer.model, train_on_gpu)
    for epoch in 1:trainer.args.max_epochs 
        losses = fit_epoch(model, trainer.opt; train_dataloader = td, val_dataloader = vd, gradient_clip_val = trainer.args.gradient_clip_val)
        
        if epoch % print_every == 0
            if !isempty(losses.val_acc)
                trainer.args.verbose && @info "Train Loss: $(losses.train_losses[end]), Val Loss: $(losses.val_losses[end]), Val Acc: $(losses.val_acc[end])"
            else
                trainer.args.verbose && @info "Train Loss: $(losses.train_losses[end]), Val Loss: $(losses.val_losses[end])"
            end
        end
        draw_metrics(model, epoch, trainer, losses)

    end
    trainer.args.verbose && display(trainer.board.plt)
    final_val_metrics = validation_step.(Ref(model), vd)
    final_val_loss = mean.(getindex.(final_val_metrics,1))
    final_val_acc = all(isnothing, getindex.(final_val_metrics, 2)) ? nothing : mean.(getindex.(final_val_metrics, 2))
    return cpu(model) , (val_loss = final_val_loss, val_acc = final_val_acc)
end

function fit_epoch(trainer::AbstractTrainer) end