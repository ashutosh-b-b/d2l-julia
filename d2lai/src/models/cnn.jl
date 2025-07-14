struct Residual{N} <: AbstractModel
    net::N
end 

function Residual(channels_in::Int; num_channels::Int = channels_in, use_1x1conv = !isequal(channels_in, num_channels), stride = 1)
    conv_chain = Chain(
            Conv((3,3) , channels_in=>num_channels, pad = 1, stride = stride),
            BatchNorm(num_channels, relu),
            Conv((3,3) , num_channels=>num_channels, pad = 1),
            BatchNorm(num_channels),
        )
    
    net = use_1x1conv ? Parallel(+, conv_chain, Conv((1,1), channels_in=>num_channels, stride = stride)) : Parallel(+, conv_chain, Flux.identity)
    Residual(net)
end

(r::Residual)(x) = relu.(r.net(x))

struct ResNetB1{N} <: AbstractModel
    net::N
end

function ResNetB1(chanel_in_size::Int=1)
    net = Chain(
        Conv((7,7), chanel_in_size => 64, pad = 3 , stride = 2),
        BatchNorm(64, relu),
        MaxPool((3,3), pad = 1, stride = 2)
    )
    ResNetB1(net)
end 
(r::ResNetB1)(x) = r.net(x)
Flux.@layer ResNetB1

struct ResNetBlock{N} <: AbstractModel 
    net::N 
end

function ResNetBlock(channel_in, num_residuals, num_channels; first_block = false)
    block = if first_block
        blocks = map(1:num_residuals) do i
            Residual(channel_in)
        end |> Chain
    else
        blocks = map(1:num_residuals) do i
            if i == 1
                return Residual(channel_in; num_channels, stride = 2)
            else
                return Residual(num_channels)
            end
        end |> Chain
    end 
    ResNetBlock(block)
end
Flux.@layer ResNetBlock
(r::ResNetBlock)(x) = r.net(x)

struct ResNet{N} <: AbstractClassifier 
    net::N
end
Flux.@layer ResNet 

function ResNet(arch::Tuple, num_classes::Int = 10, image_size = (96,96,1))
    channel_ins = last.(arch[1:end-1]) 
    net = Flux.@autosize (image_size..., 1) Chain(
        ResNetB1(image_size[3]),
        ResNetBlock(64, arch[1]..., first_block = true),
        map(arch[2:end], channel_ins) do (num_residuals, num_channels), channel_in 
            ResNetBlock(channel_in, num_residuals, num_channels)
        end |> Chain, 
        GlobalMeanPool(),
        Flux.flatten,
        Dense(_ => num_classes),
        softmax
        
    )
    ResNet(net)
end
(r::ResNet)(x) = r.net(x)