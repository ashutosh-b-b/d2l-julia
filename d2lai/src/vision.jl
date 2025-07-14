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

function offset_boxes(anchors, assigned_bb, eps = 1e-6)
    c_anc = box_corner_to_center(anchors)
    c_assigned_bb = box_corner_to_center(assigned_bb)
    offset_xy = 10 .* (c_assigned_bb[:, 1:2] - c_anc[:, 1:2]) ./ c_anc[:, 3:4]
    offset_wh = 5 * log.(eps .+ c_assigned_bb[:, 3:end] ./ c_anc[:, 3:end])
    return cat(offset_xy, offset_wh; dims = 2)
end

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