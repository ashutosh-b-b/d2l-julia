function fit_epoch(trainer::Trainer; train_dataloader = nothing , val_dataloader = nothing)
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