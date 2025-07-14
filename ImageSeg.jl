
function read_voc_images(extracted_folder; train = true)
    folder = train ? "bananas_train" : "bananas_val"
    folder_path = joinpath(extracted_folder, "banana-detection", folder)
    df = DataFrame(CSV.File(joinpath(folder_path, "label.csv")))
    img_names = df[!, 1]
    targets = df[!, 2:end] |> Array 
    targets = permutedims(targets, (2, 1))
    images = map(img_names) do img_name 
        img = Images.load(joinpath(folder_path, "images", img_name))
        img = Image(img)
        img_tensor = apply(ImageToTensor(), img) |> itemdata
        img_tensor = permutedims(img_tensor, (2,1,3))
    end
    
    images = stack(images; dims = 4)

    images, Flux.unsqueeze(targets, dims = 1) ./ 256
end

struct BananaDataset{T,V,A} <: AbstractData 
    train_data::T 
    val_data::V 
    args::A
end

function BananaDataset(; batchsize = 32)
    file = d2lai._download("banana-detection.zip")

    extracted_folder = d2lai._extract(file)

    train_data = read_data_bananas(extracted_folder; train = true)
    val_data = read_data_bananas(extracted_folder; train = false)
    args = (; extracted_folder, batchsize)
    BananaDataset(train_data, val_data, args)
end

function get_dataloader(data::BananaDataset; train = true)
    if train
        Flux.DataLoader(data.train_data; batchsize = data.args.batchsize, shuffle = true)
    else
        Flux.DataLoader(data.val_data; batchsize = data.args.batchsize)
    end
end


file = d2lai._download("VOCtrainval_11-May-2012.tar")
extracted_folder = d2lai._extract(file)

function read_voc_images(extracted_folder; train = true)
    txt_file =  train ? "train.txt" : "val.txt"
    voc_dir = joinpath(extracted_folder, "VOCdevkit/VOC2012")
    txt_fname = joinpath(voc_dir, "ImageSets", "Segmentation", txt_file)
    lines = readlines(txt_fname)
    feature_imgs = map(lines) do img_name
        img = Images.load(joinpath(voc_dir, "JPEGImages", "$img_name.jpg"))
        img = Image(img)
        img_tensor = apply(ImageToTensor(), img) |> itemdata
        img_tensor = permutedims(img_tensor, (2,1,3))
    end 
    labels = map(lines) do img_name
        img = Images.load(joinpath(voc_dir, "SegmentationClass", "$img_name.png"))
        img = Image(img)
        img_tensor = apply(ImageToTensor(), img) |> itemdata
        img_tensor = permutedims(img_tensor, (2,1,3))
    end 

end
