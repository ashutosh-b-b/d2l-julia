using d2lai, Flux, CUDA, cuDNN, Statistics, Flux.Zygote


abstract type AbstractSSDBlock <: d2lai.AbstractModel end 

struct ClassPredictor{N} <: AbstractSSDBlock
    net::N 
end
Flux.@layer ClassPredictor
(c::ClassPredictor)(x) = c.net(x)

function ClassPredictor(num_inputs::Int64, num_anchors::Int64, num_classes::Int64)
    net = Conv(
        (3,3),
        num_inputs => num_anchors * (num_classes + 1),
        pad = 1
    )

    ClassPredictor(net)
end

struct BboxPredictor{N} <: AbstractSSDBlock 
    net::N
end

function BboxPredictor(num_inputs::Int64, num_anchors::Int64)
    net = Conv(
        (3,3),
        num_inputs => num_anchors * 4, 
        pad = 1
    )
    BboxPredictor(net)
end
(c::BboxPredictor)(x) = c.net(x)

# num_anchors * (4 offsets)
function bbox_predictor(num_inputs, num_anchors)
    Conv(
        (3, 3), num_inputs => num_anchors * 4, pad = 1
    )
end

function d2lai.forward(x, block::AbstractSSDBlock)
    block(x)
end

Y1 = forward(rand(20, 20, 8, 2), ClassPredictor(8, 5, 10))
Y2 = forward(rand(10, 10, 16, 2), ClassPredictor(16, 3, 10))

@show size(Y1)
@show size(Y2)

function flatten_pred(pred)
    # shape pred: kernel_h x kernel_w x out_ch x batch_size 
    Flux.flatten(pred)
end

function concat_pred(preds)
    flatten_preds = map(preds) do pred
        flatten_pred(preds)
    end
    stack(flatten_preds, dims = 3) 
end

struct DownSampleBlk{N} <: AbstractSSDBlock
    net::N
end
Flux.@layer DownSampleBlk
(d::DownSampleBlk)(x) = d.net(x)

function DownSampleBlk(in_channels, out_channels)
    blk = []
    for _ in 1:2 
        push!(blk, Conv((3,3), in_channels => out_channels, pad = 1))
        push!(blk, BatchNorm(out_channels)),
        push!(blk, relu)
        in_channels = out_channels
    end
    net = Chain(blk..., MaxPool((2,2)))
    return DownSampleBlk(net)
end

forward(zeros(20, 20, 3, 2), DownSampleBlk(3, 10)) |> size

struct BaseNet{N} <: AbstractSSDBlock 
    net::N 
end
Flux.@layer BaseNet 
(b::BaseNet)(x) = b.net(x)

function BaseNet()
    blks = []
    num_filters = [3, 16, 32, 64]
    for i in 1:length(num_filters)-1 
        blk = DownSampleBlk(num_filters[i], num_filters[i+1])
        push!(blks, blk)
    end
    return BaseNet(Chain(blks...))
end

@info size(BaseNet()(rand(256, 256, 3, 2)))

function get_blk(i)
    blk = if i == 1
        blk = BaseNet()
    elseif i == 2
        blk = DownSampleBlk(64, 128)
    elseif i == 5
        blk = GlobalMaxPool()
    else
        blk = DownSampleBlk(128, 128)
    end
    return blk
end

function blk_forward(X, blk, sz, ratio, cls_predictor, bbox_predictor)
    Y = blk(X)
    anchors = multibox_prior(Y, sz, ratio)
    cls_preds = cls_predictor(Y)
    blk_preds = bbox_predictor(Y)
    return Y, anchors, cls_preds, blk_preds
end

sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
         [0.88, 0.961]]
ratios = repeat([[1, 2, 0.5]], 5)
num_anchors = length(sizes[1]) + length(ratios[1]) - 1

struct TinySSD{B, CP, BP, A} <: AbstractClassifier 
    blocks::B 
    class_predictors::CP 
    bbox_predictors::BP
    args::A
end 
Flux.@layer TinySSD trainable=(blocks, class_predictors, bbox_predictors)

function TinySSD(num_classes; kw...)
    idx_to_in_channels = [64, 128, 128, 128, 128]
    blocks = map(1:5) do i 
        get_blk(i) 
    end
    class_predictors = map(1:5) do i 
        ClassPredictor(idx_to_in_channels[i], num_anchors, num_classes)
    end
    bbox_predictors = map(1:5) do i 
        BboxPredictor(idx_to_in_channels[i], num_anchors)
    end
    TinySSD(blocks, class_predictors, bbox_predictors, (; num_classes, kw...))
end

function (model::TinySSD)(x::AbstractArray)
    blocks, class_predictors, bbox_predictors = model.blocks, model.class_predictors, model.bbox_predictors
    sizes, ratios = model.args.sizes, model.args.ratios 
    anchors = AbstractArray{<:Real}[]
    cls_preds =  AbstractArray{<:Real}[]
    bbox_preds =  AbstractArray{<:Real}[]
    batch_size = size(x)[end]
    for i in 1:length(blocks)
        blk, class_predictor, bbox_predictor, sz, rt = blocks[i], class_predictors[i], bbox_predictors[i], sizes[i], ratios[i]
        x, anchors_i, cls_preds_i, bbox_preds_i = blk_forward(x, blk, sz, rt, class_predictor, bbox_predictor)
        anchors = [anchors; [anchors_i]]
        cls_preds = [cls_preds; [reshape(permutedims(cls_preds_i, (3, 2, 1, 4)), model.args.num_classes+1, :, batch_size)]]
        bbox_preds = [bbox_preds; [reshape(permutedims(bbox_preds_i, (3, 2, 1, 4)), :, batch_size)]]
    end
    
    anchors = reduce(vcat, anchors)
    # cls_preds = reduce(vcat, cls_preds)
    cls_preds = reduce((a...) -> cat(a..., dims = 2), cls_preds)
    bbox_preds = reduce(vcat, bbox_preds)
    anchors, cls_preds, bbox_preds
end

function cls_loss(model::TinySSD, y::AbstractArray, y_pred::AbstractArray)
    loss = Flux.Losses.logitcrossentropy(y_pred, Flux.onehotbatch(y, 0:model.args.num_classes); agg = identity)
    # loss = dropdims(loss, dims = 1)
end
function bbox_loss(args...)
    loss = Flux.Losses.mae(args...; agg = identity) 
    # mean(loss; dims = 1)
end

# function calc_loss(model::TinySSD, cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
#     class_loss = cls_loss(model, Int.(cls_labels), cls_preds)
#     bounding_box_loss = bbox_loss(bbox_labels .* bbox_masks, bbox_preds .* bbox_masks)
#     loss = class_loss + bounding_box_loss
#     mean(loss)
# end
function calc_loss(model::TinySSD, cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
    batch_size = size(cls_preds, 3)
    num_anchors = size(cls_preds, 2)
    num_classes = size(cls_preds, 1)

    # Reshape to match PyTorch: (batch * anchors, num_classes)
    cls_preds_flat = reshape(cls_preds, num_classes, :) 
    cls_labels_flat = vec(cls_labels)
    cls_loss_flat = cls_loss(model, Int.(cls_labels_flat), cls_preds_flat)
    cls_loss_per_image = reshape(cls_loss_flat, num_anchors, batch_size)
    cls_loss_mean = mean(cls_loss_per_image; dims=1)  # shape: (1, batch_size)

    # # bbox loss: already (anchors, 4, batch)

    bbox_loss_raw = bbox_loss(bbox_preds.* bbox_masks, bbox_labels.* bbox_masks)
    bbox_loss_per_image = mean(bbox_loss_raw; dims=1)  # shape: (1, batch_size)

    total_loss_per_image = cls_loss_mean + bbox_loss_per_image
    mean(total_loss_per_image)
    # return dropdims(total_loss_per_image; dims=1)  # shape: (batch_size,)
end

function cls_eval(cls_preds, cls_labels)
    arg_ = getindex.(argmax(cls_preds; dims = 1), 1)
    sum(dropdims(arg_, dims = 1) .- 1 .== cls_labels) / length(cls_labels)
end

function bbox_eval(bbox_preds, bbox_labels, bbox_masks)
    sum(abs.((bbox_labels - bbox_preds) .* bbox_masks)) / length(bbox_labels)
end


model = TinySSD(1; sizes, ratios) |> f64 |> gpu


data = d2lai.BananaDataset(; batchsize = 32)


# @testset "multibox target" begin 
#     ground_truth = [
#         0    0.1   0.08  0.52  0.92;
#         1    0.55  0.2   0.9   0.88
#     ] |> gpu

#     anchors_test = [
#         0.0   0.1   0.2   0.3;
#         0.15  0.2   0.4   0.4;
#         0.63  0.05  0.88  0.98;
#         0.66  0.45  0.8   0.8;
#         0.57  0.3   0.92  0.9
#     ] |> gpu
#     labels = multibox_target(
#         Flux.unsqueeze(anchors_test, dims = 3), 
#         Flux.unsqueeze(ground_truth, dims = 3), 
#         gpu)
# end
out =  multibox_target(anchors_, d_[2], gpu; iou_threshold = 0.5)
mapp, la = getindex.(out,1), getindex.(out, 2)
let 

    d_ = d2lai.get_dataloader(data) |> collect |> first |> gpu
    current_loss, gs = Zygote.withgradient(model) do m
        anchors_, cls_preds, bbox_preds = model(d_[1]);
        bbox_labels, bbox_masks, cls_labels = Zygote.@ignore multibox_target(anchors_, d_[2], gpu; iou_threshold = 0.3)
        loss = calc_loss(model, cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
    end
end

trainer = Trainer(model, data, Adam(0.1); max_epochs = 20)
train_iter = d2lai.get_dataloader(data) |> gpu
# model = model
state = Flux.setup(Adam(0.05), model)
for i in 1:20
    losses = (loss = [], class_error = [], bbox_error = [])
    for d in train_iter
        current_loss, gs = Zygote.withgradient(model) do m
            anchors_, cls_preds, bbox_preds = m(d[1]);
            bbox_labels, bbox_masks, cls_labels = Zygote.@ignore multibox_target(anchors_, d[2], gpu; iou_threshold = 0.5)
            loss = calc_loss(m, cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
        end
        Flux.Optimise.update!(state, model, gs[1])
        push!(losses.loss, current_loss)
        anchors_, cls_preds, bbox_preds = model(d[1]);
        bbox_labels, bbox_masks, cls_labels = multibox_target(anchors_, d[2], gpu; iou_threshold = 0.5)
        class_error = cls_eval(cls_preds, cls_labels)
        bbox_error = bbox_eval(bbox_preds, bbox_labels, bbox_masks)
        push!(losses.class_error, 1 - class_error)
        push!(losses.bbox_error, bbox_error)
    end
    @info "Epoch : $i, Loss: $(mean(losses.loss)), Class Err: $(mean(losses.class_error)), BBOX Err: $(mean(losses.bbox_error))"
end

function train_model(trainer::Trainer)
    train_iter = d2lai.get_dataloader(trainer.data) |> gpu
    model = trainer.model
    state = Flux.setup(trainer.opt, model)
    for i in 1:trainer.args.max_epochs
        losses = (loss = [], class_error = [], bbox_error = [])
        for d_ in train_iter
            current_loss, gs = Zygote.withgradient(model) do m
                anchors_, cls_preds, bbox_preds = m(d_[1]);
                bbox_labels, bbox_masks, cls_labels = Zygote.@ignore multibox_target(anchors_, d_[2], gpu; iou_threshold = 0.5)
                loss = calc_loss(m, cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
            end
            Flux.Optimise.update!(state, model, gs[1])
            push!(losses.loss, current_loss)
            anchors_, cls_preds, bbox_preds = model(d_[1]);
            bbox_labels, bbox_masks, cls_labels = multibox_target(anchors_, d_[2], gpu; iou_threshold = 0.5)
            class_error = cls_eval(cls_preds, cls_labels)
            bbox_error = bbox_eval(bbox_preds, bbox_labels, bbox_masks)
            push!(losses.class_error, 1 - class_error)
            push!(losses.bbox_error, bbox_error)
        end
        @info "Epoch : $i, Loss: $(mean(losses.loss)), Class Err: $(mean(losses.class_error)), BBOX Err: $(mean(losses.bbox_error))"
    end
    model
end



model = train_model(trainer);


function predict(model::TinySSD, X)
    Flux.testmode!(model)  # equivalent to net.eval()

    anchors, cls_preds, bbox_preds = model(X)  # assume batch-last layout
    cls_probs = softmax(cls_preds; dims=1)     # softmax over class dimension

    output = multibox_detection(cls_probs, bbox_preds, anchors) |> cpu

    # Output is a Vector of Matrices (one per image)
    preds = output[1]  # predictions for first image

    # Keep only non-background predictions (class_id != -1)
    valid_idx = findall(preds[:, 1] .!= -1)

    return preds[valid_idx, :]
end

function nms(boxes, scores, iou_threshold)
    B = sortperm(scores; rev=true) |> cpu
    boxes = boxes |> cpu
    keep = []
    keep_idx = Int[]

    while !isempty(B)
        i = B[1]
        push!(keep_idx, i)
        length(B) == 1 && break
        ious = d2lai.box_iou(view(boxes, i:i, :), view(boxes, B[2:end], :))
        ious_vec = vec(ious)
        mask = ious_vec .<= iou_threshold
        B = B[findall(mask) .+ 1]  # skip current top index
    end

    return keep_idx
end

function offset_inverse(anchors, offset_pred)
    anc = d2lai.box_corner_to_center(anchors)  # (num_anchors, 4)
    pred_xy = (offset_pred[:, 1:2] .* anc[:, 3:4]) ./ 10.0 .+ anc[:, 1:2]
    pred_wh = exp.(offset_pred[:, 3:4] ./ 5.0) .* anc[:, 3:4]
    pred_center = hcat(pred_xy, pred_wh)
    return d2lai.boxes_center_to_corner(pred_center)
end
function multibox_detection(cls_probs, offset_preds, anchors; nms_threshold = 0.5, pos_threshold =1e-6)
    device = isa(cls_probs, CuArray) ? gpu : cpu 
    num_anchors = size(anchors, 1)
    batch_size = size(cls_probs)[end]
    out = map(1:batch_size) do b 
        cls_prob, offset_pred = cls_probs[:, :, b], reshape(offset_preds[:, b], 4, :)
        offset_pred = permutedims(offset_pred, (2,1))
        cls_prob_fg = cls_prob[2:end, :]  # remove background (class 0)
        conf, class_id = maximum(cls_prob_fg; dims = 1), getindex.(argmax(cls_prob_fg, dims = 1), 1) .+ 1
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, vec(conf), 0.5) |> cpu
        all_idx = collect(1:num_anchors) |> cpu 
        combined = vcat(keep, all_idx) |> cpu
        counts = countmap(combined)
        non_keep = [i for (i, c) in counts if c == 1]

        # Order: keep first, then non-keep
        all_sorted = vcat(keep, non_keep)
        class_out = copy(class_id)
        class_out[non_keep] .= -1  # background
        class_out = class_out[all_sorted]
        conf_out = conf[all_sorted]
        bb_out = predicted_bb[all_sorted, :]

        # Threshold low confidence predictions
        low_conf_mask = conf_out .< pos_threshold
        class_out[low_conf_mask] .= -1
        conf_out[low_conf_mask] .= 1 .- conf_out[low_conf_mask]

        hcat(Float32.(class_out), conf_out, bb_out)

    end
    
end

output = predict(model, X)
img = load("./Julia_Notebooks/img/banana.jpg")
img = Image(img)
img_tensor = apply(ImageToTensor(), img) |> itemdata
img_tensor = permutedims(img_tensor, (2,1,3))
img_tensor = Flux.unsqueeze(img_tensor, dims = 4)

model = model |> cpu

output = predict(model, img_tensor)

img = load("./Julia_Notebooks/img/banana.jpg")

function display(img, output, threshold)
    plt = plot(img)
    for r in eachrow(output)
        score = r[2]
        if score > threshold 
            h, w = size(img)
            bbox = reshape(r[3:6] .* 256, 1, 4)
            @info size(bbox)
            plt = d2lai.show_bboxes(plt, bbox; colors = [:white])
        else
            continue 
        end
    end
    plt
end
display(img, output, 0.2)

