# function fit_epoch(trainer::Trainer; train_dataloader = nothing , val_dataloader = nothing)
#     losses = (train_losses = [], val_losses = [])
#     for batch in train_dataloader
#         gs = gradient(Flux.Params([model.w, model.b])) do 
#             training_step(trainer.model, batch)
#         end 
#         step!(trainer, gs)
#         train_loss = training_step(trainer.model, batch)
#         push!(losses.train_losses, train_loss)
#     end
#     for batch in val_dataloader
#         loss = validation_step(trainer.model, batch)
#         push!(losses.val_losses , loss)
#     end
#     return losses
# end

function clip_gradients! end

function d2lai.clip_gradients!(gs, gradient_clip_val, model)
    sums = fmap(gs, walk=Flux.Functors.IterateWalk()) do g
       !isnothing(g) && sum(g.^2)
   end
   norm = sqrt(sum(filter(x -> x != false, collect(sums))))
   g_ = fmap(gs) do d
       if !isnothing(d)
           d = d.* (1. / norm)
       end
   end
   g_
end

function fit_epoch(model::AbstractClassifier, opt; train_dataloader = nothing, val_dataloader = nothing, gradient_clip_val = 0.)
    losses = (train_losses = [], val_losses = [], val_acc = [])
    state = Flux.setup(opt, model)
    for batch in train_dataloader
        gs = gradient(model) do m
            training_step(m, batch)
        end
        gs = if gradient_clip_val > 0. 
            clip_gradients!(gs, gradient_clip_val, model)
        else
            gs
        end
        Flux.update!(state, model, gs[1])
        train_loss = training_step(model, batch)
        push!(losses.train_losses, train_loss)
    end
    for batch in val_dataloader
        loss, acc = validation_step(model, batch)
        push!(losses.val_losses , loss)
        !isnothing(acc) && push!(losses.val_acc, acc)
    end
    return losses
end

function fit_epoch(model::AbstractModel, opt; train_dataloader = nothing, val_dataloader = nothing, gradient_clip_val = 0.)
    losses = (train_losses = [], val_losses = [], val_acc = [])
    state = Flux.setup(opt, model)
    for batch in train_dataloader
        gs = gradient(model) do m
            training_step(m, batch)
        end
        gs = if gradient_clip_val > 0. 
            clip_gradients!(gs, gradient_clip_val, model)
        else
            gs
        end
        Flux.Optimise.update!(state, model, gs[1])
        train_loss = training_step(model, batch)
        push!(losses.train_losses, train_loss)
    end
    for batch in val_dataloader
        loss, acc = validation_step(model, batch)
        push!(losses.val_losses , loss)
    end
    return losses
end