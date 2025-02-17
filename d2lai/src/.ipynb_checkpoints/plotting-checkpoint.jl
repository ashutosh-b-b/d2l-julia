struct ProgressBoard
    xlabel::Union{Nothing, String}
    xscale::Symbol
    yscale::Symbol
    colors::Vector{Any}
    size::Tuple
    plt::Plots.Plot
    labels_to_idx::AbstractDict
    anim::Animation
    function ProgressBoard(xlabel, y_labels = []; xscale = :identity, yscale = :identity, colors = [:blue, :red, :orange], size = (25, 25))
        plt = plot(fill([], length(y_labels)), fill([], length(y_labels)), xlabel = xlabel, xscale = xscale, yscale= yscale, labels  = y_labels)
        labels_to_idx = Dict(vec(y_labels) .=> collect(1:length(y_labels)))
        new(xlabel, xscale, yscale, colors, size, plt, labels_to_idx, Animation())
    end    
end

function draw(board::ProgressBoard, x::Number, y::Number, label::String)
    if haskey(board.labels_to_idx, label) 
        idx = board.labels_to_idx[label]
        push!(board.plt, idx[1], x, y)
    else
        idx = length(keys(board.labels_to_idx)) + 1
        board.labels_to_idx[labels] = idx
        plot!(board.plt, x, y, label = label, xlabel = board.xlabel)
    end
    frame(board.anim)
end

function draw(board::ProgressBoard, x::AbstractVector, y::AbstractVector, label::String)
    @assert length(x) == length(y) "Length `x` doesnot match length of `y`"
    if haskey(board.labels_to_idx, label) 
        idx = board.labels_to_idx[label]
        push!.(board.plt[idx], x, y)
    else
        idx = length(keys(board.labels_to_idx)) + 1
        board.labels_to_idx[labels] = idx
        plot!(board.plt, x, y, label = label, xlabel = board.xlabel)
    end
    frame(board.anim)
end

function save_plot(board::ProgressBoard, file_name = "plot_1.png")
    savefig(board.plt, file_name)
end

function save_gif(board::ProgressBoard, file_name = "anim_fps.gif"; fps = 15)
    gif(board.anim, file_name, fps = fps)
end

function draw_model(step, i, epoch, n_batches, label = "train"; acc = false)
    if !acc
        loss = step(model, batch, loss)
        draw(board, epoch + i/n_batches, loss, label + "_loss")
    else
        loss, acc = step(model, batch, loss)
        draw(board, epoch + i/n_batches, loss, label + "_loss")
        draw(board, epoch + i/n_batches, acc, label + "_acc")
    end
end