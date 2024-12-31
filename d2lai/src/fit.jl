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

function fit_epoch(model::AbstractModel, opt; train_dataloader = nothing, val_dataloader = nothing)
    losses = (train_losses = [], val_losses = [], val_acc = [])
    for batch in train_dataloader
        ps = Flux.params(model)
        gs = gradient(ps) do 
            training_step(model, batch)
        end
        Flux.Optimise.update!(opt, ps, gs)
        train_loss = training_step(model, batch)
        push!(losses.train_losses, train_loss)
    end
    for batch in val_dataloader
        loss, acc = validation_step(model, batch)
        push!(losses.val_losses , loss)
        push!(losses.val_acc, acc)
    end
    return losses
end