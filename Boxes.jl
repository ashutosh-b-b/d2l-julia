using Images, Plots
function box_corner_to_center(boxes)
    x1, y1, x2, y2 = boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
    cx = (x1 .+ x2) ./ 2
    cy = (y1 .+ y2) ./ 2
    w = x2 .- x1
    h = y2 .- y1
    boxes = stack((cx, cy, w, h), dims = 2)
    return boxes 
end

function boxes_center_to_corner(boxes)
    cx, cy, w, h =  boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = stack((x1, y1, x2, y2), dims = 2)
    return boxes
end

dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0]
boxes = stack((dog_bbox, cat_bbox), dims = 1)
boxes_center_to_corner(box_corner_to_center(boxes)) == boxes

rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])

function bbox_to_rect(plt, bbox, color, label = nothing)
    x1, y1, x2, y2 = bbox
    rect = rectangle(x2-x1, y2-y1, x1, y1)
    if !isnothing(label) 
        plot!(plt, rect, fillalpha=0, linecolor = color, label = label)
    else
        plot!(plt, rect, fillalpha=0, linecolor = color)
    end
    plt
end

img = load("./Julia_Notebooks/img/catdog.jpg")

plt = plot(img)

bbox_to_rect(plt, dog_bbox, :red)
bbox_to_rect(plt, cat_bbox, :blue)

function multibox_prior(data, sizes, ratio)
    device = isa(data, CuArray) ? gpu : cpu
    in_height, in_width = size(data)[1:2]
    num_sizes, num_ratios = length(sizes), length(ratio)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # Scaled steps in y axis
    steps_w = 1.0 / in_width  # Scaled steps in x axis

    center_h = (collect(0:in_height-1) .+ offset_h).*steps_h
    center_w = (collect(0:in_width-1) .+ offset_w).*steps_w

    shift_y, shift_x = center_h' .* ones(length(center_w)), ones(length(center_h))' .* center_w
    shift_y, shift_x = vec(shift_y), vec(shift_x)
    
    w = vcat(
        sizes .* sqrt.(ratio[1:1]), sizes[1:1] .* sqrt.(ratio[2:end])
    )
    h = vcat(
        sizes ./ sqrt.(ratio[1:1]), sizes[1:1] ./ sqrt.(ratio[2:end])
    )
    anchor_manipulations = stack([-w, -h, w, h]) ./ 2
    anchor_manipulations = repeat(anchor_manipulations, in_height*in_width, 1)
    out_grid = stack([shift_x, shift_y, shift_x, shift_y])
    out_grid = repeat(out_grid, inner = (boxes_per_pixel, 1)) |> device
    Y = out_grid .+ anchor_manipulations
    Flux.unsqueeze(Y, dims = 3)
end

multibox_prior(rand(2,2), [0.5], [1., 2.])

h, w = size(img) # 561, 728

print(h, w)
X = rand(h, w, 3, 1) |> gpu  # Construct input data 
Y = multibox_prior(X, gpu([0.75, 0.5, 0.25]), gpu([1, 2, 0.5]))
size(Y)

# boxes = reshape(Y, h, w, 5, 4)
# boxes[250, 250, :, :]

Y_reshaped = reshape(Y, 5, h, w, 4)
Y_reshaped = permutedims(Y_reshaped, (2, 3, 1, 4)) 


function show_bboxes(plt, bboxes; labels = nothing, colors = [])
    default_colors = [:blue, :green, :red, :yellow, :orange]
    colors = isempty(colors) ? default_colors : colors
    for (i, bbox) in enumerate(eachslice(bboxes, dims = 1))
        color = colors[i % length(colors) + 1]
        plt = if isnothing(labels)
            bbox_to_rect(plt, bbox, color)
        else
            bbox_to_rect(plt, bbox, color, labels[i])
        end
    end
    plt
end

plt = plot(img)

show_bboxes(plt, 
    Y_reshaped[250, 250, :, :] .* reshape([w, h, w, h], 1, :);
    labels = (
        "s=0.75, r=1", "s=0.5, r=1", "s=0.25, r=1", "s=0.75, r=2",
             "s=0.75, r=0.5")
)



function box_iou(boxes1, boxes2)
    # Helper function: area = (xmax - xmin) * (ymax - ymin)
    function box_area(boxes)
        (boxes[:, 3] .- boxes[:, 1]) .* (boxes[:, 4] .- boxes[:, 2])
    end

    areas1 = box_area(boxes1)  # shape: (n1,)
    areas2 = box_area(boxes2)  # shape: (n2,)

    n1 = size(boxes1, 1)
    n2 = size(boxes2, 1)

    inter_top_left = max.(
        reshape(boxes1[:, 1:2], n1, 1, 2),   # broadcast to (n1, n2, 2)
        reshape(boxes2[:, 1:2], 1, n2, 2)
    )
    inter_bot_right = min.(
        reshape(boxes1[:, 3:4], n1, 1, 2),
        reshape(boxes2[:, 3:4], 1, n2, 2)
    )

    inter_wh = max.(inter_bot_right .- inter_top_left, 0.0f0)  # (n1, n2, 2)
    inter_areas = inter_wh[:, :, 1:1] .* inter_wh[:, :, 2:2]       # (n1, n2)

    union_areas = reshape(areas1, :, 1) .+ reshape(areas2, 1, :) .- inter_areas

    return inter_areas ./ union_areas  # shape: (n1, n2)
end

function assign_anchor_to_bbox(ground_truth, anchors, iou_threshold; device = cpu)
    num_anchors, num_gt_boxes = size(anchors, 1), size(ground_truth, 1)
    jaccard = box_iou(anchors, ground_truth)
    anchors_bbox_map = fill(-1., num_anchors) |> device
    max_ious, indices = findmax(jaccard, dims = 2)
    max_ious = vec(max_ious)
    indices = getindex.(indices, 2)
    anc_i = findall(max_ious .>= iou_threshold)
    box_j = indices[max_ious .>= iou_threshold]
    anchors_bbox_map[anc_i] .= box_j
    col_discard = fill(-1, num_anchors)
    row_discard = fill(-1, num_gt_boxes)
    for _ in 1:num_gt_boxes
        max_idx = argmax(jaccard)
        anc_idx = max_idx[1]
        box_idx = max_idx[2]
        anchors_bbox_map[anc_idx:anc_idx] = box_idx
        jaccard[:, box_idx:box_idx] = col_discard
        jaccard[anc_idx:anc_idx, :] = row_discard
    end
    anchors_bbox_map
end



jaccard = box_iou(gpu(anchors), gpu(ground_truth[:, 2:end]))

max_ious, indices = findmax(jaccard, dims = 2)

findall(max_ious .>= 0.5)

indices_of_max_ious = argmax(jaccard, dims = 2)
indices_of_max_ious[max_ious .>= 0.5]
anc = assign_anchor_to_bbox(gpu(ground_truth[:, 2:end]), gpu(anchors), 0.5; device = gpu) .|> Int



function offset_boxes(anchors, assigned_bb, eps = 1e-6)
    c_anc = box_corner_to_center(anchors)
    c_assigned_bb = box_corner_to_center(assigned_bb)
    offset_xy = 10 .* (c_assigned_bb[:, 1:2] - c_anc[:, 1:2]) ./ c_anc[:, 3:4]
    offset_wh = 5 * log.(eps .+ c_assigned_bb[:, 3:end] ./ c_anc[:, 3:end])
    return cat(offset_xy, offset_wh; dims = 2)
end
multibox_target(Flux.unsqueeze(anchors, dims = 3), Flux.unsqueeze(ground_truth, dims = 3))

function multibox_target(anchors, labels, device = cpu; iou_threshold = 0.5)
    batch_size = size(labels) |> last 
    anchors = dropdims(anchors; dims = 3)
    num_anchors = size(anchors, 1)
    out = map(1:batch_size) do i 
        label = labels[:, :, i]
        gt_labels = label[:, 1]
        # assigns each anchor to a ground truth bounding box 
        anchors_bbox_map = assign_anchor_to_bbox(label[:, 2:end], anchors, iou_threshold; device = device) .|> Int
        # the class for each anchor is basically the class of assigned bounding box. 
        # If the box is not assigned, the class is 0.
        
        # assign zeros of size num anchors by default
        anchor_box_classes = zeros(num_anchors) |> device
        # lhs: get elements of the preallocated array only for the anchors that are assigned
        # rhs: get elements of gt_labels for bounding boxes that are assigned to anchors 
        anchor_box_classes[anchors_bbox_map .> 0] = gt_labels[anchors_bbox_map[anchors_bbox_map .> 0]] .+ 1
        
        assigned_bbox = zeros(num_anchors, 4) |> device 
        assigned_bbox[anchors_bbox_map .> 0, :] = label[anchors_bbox_map[anchors_bbox_map .> 0], 2:end]

        bbox_mask = reshape(Int.(anchors_bbox_map .>= 0), num_anchors, 1)
        offset = offset_boxes(anchors, assigned_bbox) .* bbox_mask
        offset = reduce(vcat, eachrow(offset))
        offset, repeat(bbox_mask, inner = (4,1)), anchor_box_classes
    end
    offset, assigned_bbox, class_labels = getindex.(out, 1), getindex.(out, 2), getindex.(out, 3)
    reduce(hcat, offset), reduce(hcat, assigned_bbox), reduce(hcat, class_labels)
end

gt = gpu(ground_truth)
label = gt[:, 1][anc[2]:anc[2]]

map(anc) do a 
    if a > 0
        label[a:a]
    end
end
mapings = multibox_target(
    gpu(Flux.unsqueeze(anchors_, dims = 3)), 
    gpu(Flux.unsqueeze(ground_truth, dims = 3)), gpu)

mapings = multibox_target(
    gpu(anchors_), 
    gpu(d[2]), gpu)
offset, assigned_bbox, class_labels = mapings
offset_boxes(anchors, assigned_bbox)

using d2lai
using Flux, Images, Plots, DataFrames, CSV
using DataAugmentation

file = d2lai._download("banana-detection.zip")

extracted_folder = d2lai._extract(file)



data = BananaDataset(; batchsize = 32)

train_iter = get_dataloader(data)

d = first(train_iter)

# num_anchors * classes for each anchor box (add + 1 for background)
using d2lai, Flux 
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

    out = map(blocks, class_predictors, bbox_predictors, sizes, ratios) do blk, class_predictor, bbox_predictor, size, ratio
        x, anchors_i, cls_preds_i, bbox_preds_i = blk_forward(x, blk, size, ratio, class_predictor, bbox_predictor)
    end
    x = getindex.(out, 1)
    anchors, cls_preds, bbox_preds = getindex.(out, 2), getindex.(out, 3), getindex.(out, 4)
    anchors = reduce(vcat, anchors)
    cls_preds = reduce(vcat, Flux.flatten.(cls_preds))
    cls_preds = reshape(cls_preds, model.args.num_classes + 1, :, size(cls_preds, 2))
    bbox_preds = reduce(vcat, Flux.flatten.(bbox_preds))
    anchors, cls_preds, bbox_preds
end

model = TinySSD(1; sizes, ratios) |> f64 |> gpu

gs = Zygote.gradient(model) do m
    anchors_, cls_preds, bbox_preds = m(gpu(d[1]));
    Zygote.@ignore bbox_labels, bbox_masks, cls_labels = multibox_target(anchors_, gpu(d[2]), gpu)
    loss = calc_loss(m, cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
end
trainer = Trainer(model, data, opt; max_epochs = 10)

function cls_loss(model::TinySSD, y::AbstractArray, y_pred::AbstractArray{<:Int})
    loss = Flux.Losses.logitcrossentropy(y, Flux.onehotbatch(y_pred, 0:model.args.num_classes); agg = identity)
    loss = dropdims(loss, dims = 1)
    mean(loss; dims = 1)
end
function bbox_loss(args...)
    loss = Flux.Losses.mae(args...; agg = identity) 
    mean(loss; dims = 1)
end

function calc_loss(model::TinySSD, cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
    class_loss = cls_loss(model, cls_preds, Int.(cls_labels))
    bounding_box_loss = bbox_loss(bbox_labels .* bbox_masks, bbox_preds .* bbox_masks)
    loss = class_loss + bounding_box_loss
    mean(loss)
end

state = Flux.setup(opt, model)

for epoch in 1:trainer.max_epochs 
    for (X, y) in train_iter 
        gs = Zygote.gradient(model) do m
            anchors, cls_preds, bbox_preds = model(X)
            Zygote.@ignore bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, y, gpu)
            loss = calc_loss(model, cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
        end
        Flux.Optimise.update!(state, model, gs[1])
    end
end

img = colorview(RGB, permutedims(data.train_data[1][:, :, :, 1], (3, 2, 1)))

plt = plot(img)

d2lai.show_bboxes(plt, data.train_data[2][:, 2:end, 1].*256)