# Networks Using Blocks (VGG)
:label:`sec_vgg`

While AlexNet offered empirical evidence that deep CNNs
can achieve good results, it did not provide a general template
to guide subsequent researchers in designing new networks.
In the following sections, we will introduce several heuristic concepts
commonly used to design deep networks.

Progress in this field mirrors that of VLSI (very large scale integration) 
in chip design
where engineers moved from placing transistors
to logical elements to logic blocks :cite:`Mead.1980`.
Similarly, the design of neural network architectures
has grown progressively more abstract,
with researchers moving from thinking in terms of
individual neurons to whole layers,
and now to blocks, repeating patterns of layers. A decade later, this has now
progressed to researchers using entire trained models to repurpose them for different, 
albeit related, tasks. Such large pretrained models are typically called 
*foundation models* :cite:`bommasani2021opportunities`. 

Back to network design. The idea of using blocks first emerged from the
Visual Geometry Group (VGG) at Oxford University,
in their eponymously-named *VGG* network :cite:`Simonyan.Zisserman.2014`.
It is easy to implement these repeated structures in code
with any modern deep learning framework by using loops and subroutines.


```julia
using Pkg; Pkg.activate("../../d2lai")
using d2lai
using Flux 
using CUDA, cuDNN
```

    [32m[1m  Activating[22m[39m project at `/workspace/d2l-julia/d2lai`
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mPrecompiling d2lai [749b8817-cd67-416c-8a57-830ea19f3cc4] (cache misses: include_dependency fsize change (2))


## (**VGG Blocks**)
:label:`subsec_vgg-blocks`

The basic building block of CNNs
is a sequence of the following:
(i) a convolutional layer
with padding to maintain the resolution,
(ii) a nonlinearity such as a ReLU,
(iii) a pooling layer such
as max-pooling to reduce the resolution. One of the problems with 
this approach is that the spatial resolution decreases quite rapidly. In particular, 
this imposes a hard limit of $\log_2 d$ convolutional layers on the network before all 
dimensions ($d$) are used up. For instance, in the case of ImageNet, it would be impossible to have 
more than 8 convolutional layers in this way. 

The key idea of :citet:`Simonyan.Zisserman.2014` was to use *multiple* convolutions in between downsampling
via max-pooling in the form of a block. They were primarily interested in whether deep or 
wide networks perform better. For instance, the successive application of two $3 \times 3$ convolutions
touches the same pixels as a single $5 \times 5$ convolution does. At the same time, the latter uses approximately 
as many parameters ($25 \cdot c^2$) as three $3 \times 3$ convolutions do ($3 \cdot 9 \cdot c^2$). 
In a rather detailed analysis they showed that deep and narrow networks significantly outperform their shallow counterparts. This set deep learning on a quest for ever deeper networks with over 100 layers for typical applications.
Stacking $3 \times 3$ convolutions
has become a gold standard in later deep networks (a design decision only to be revisited recently by 
:citet:`liu2022convnet`). Consequently, fast implementations for small convolutions have become a staple on GPUs :cite:`lavin2016fast`. 

Back to VGG: a VGG block consists of a *sequence* of convolutions with $3\times3$ kernels with padding of 1 
(keeping height and width) followed by a $2 \times 2$ max-pooling layer with stride of 2
(halving height and width after each block).
In the code below, we define a function called `vgg_block`
to implement one VGG block.

The function below takes two arguments,
corresponding to the number of convolutional layers `num_convs`
and the number of output channels `num_channels`.



```julia
struct VGGBlock{N} <: AbstractModel 
    net::N
end

function VGGBlock(num_convs, in_channels, out_channels, conv_size = 3)
    layers = map(1:num_convs) do i 
        if i == 1
            return Conv((conv_size,conv_size), in_channels => out_channels, relu, pad = 1)
        else
            return Conv((conv_size,conv_size), out_channels => out_channels, relu, pad = 1)
        end
    end
    net = Chain(layers..., 
        MaxPool((2,2), stride = 2),
    )
    VGGBlock(net)
end

Flux.@layer VGGBlock 

(v::VGGBlock)(x) = v.net(x)
```

# [**VGG Network**]
:label:`subsec_vgg-network`

Like AlexNet and LeNet, 
the VGG Network can be partitioned into two parts:
the first consisting mostly of convolutional and pooling layers
and the second consisting of fully connected layers that are identical to those in AlexNet. 
The key difference is 
that the convolutional layers are grouped in nonlinear transformations that 
leave the dimensonality unchanged, followed by a resolution-reduction step, as 
depicted in :numref:`fig_vgg`. 

![From AlexNet to VGG. The key difference is that VGG consists of blocks of layers, whereas AlexNet's layers are all designed individually.](../img/vgg.svg)
:width:`400px`
:label:`fig_vgg`

The convolutional part of the network connects several VGG blocks from :numref:`fig_vgg` (also defined in the `vgg_block` function)
in succession. This grouping of convolutions is a pattern that has 
remained almost unchanged over the past decade, although the specific choice of 
operations has undergone considerable modifications. 
The variable `arch` consists of a list of tuples (one per block),
where each contains two values: the number of convolutional layers
and the number of output channels,
which are precisely the arguments required to call
the `vgg_block` function. As such, VGG defines a *family* of networks rather than just 
a specific manifestation. To build a specific network we simply iterate over `arch` to compose the blocks.



```julia
struct VGGNet{N} <: AbstractClassifier 
    net::N 
end

function VGGNet(arch::Tuple, num_classes = 10)
    out_channels = getindex.(arch, 2)
    in_channels = (1, out_channels[1:end-1]...)
    num_convs = getindex.(arch, 1)
    blocks = map(num_convs, out_channels, in_channels) do n_conv, out_ch, in_ch
        VGGBlock(n_conv, in_ch, out_ch)
    end
    net = Flux.@autosize (224, 224, 1, 1) Chain(
        blocks...,
        Flux.flatten,
        Dense(_ => 4096, relu),
        Dropout(0.5),
        Dense(4096 => 4096, relu),
        Dropout(0.5),
        Dense(4096 => num_classes),
        softmax,
    )
    VGGNet(net)
end

Flux.@layer VGGNet

(vnet::VGGNet)(x) = vnet.net(x)

```

The original VGG network had five convolutional blocks, among which the first two have one convolutional layer each and the latter three contain two convolutional layers each. The first block has 64 output channels and each subsequent block doubles the number of output channels, until that number reaches 512. Since this network uses eight convolutional layers and three fully connected layers, it is often called VGG-11.




```julia
VGGNet(((1, 64), (1, 128), (2, 256), (2, 512), (2, 512)))
```




    VGGNet(
      Chain(
        VGGBlock(
          Chain(
            Conv((3, 3), 1 => 64, relu, pad=1),  [90m# 640 parameters[39m
            MaxPool((2, 2)),
          ),
        ),
        VGGBlock(
          Chain(
            Conv((3, 3), 64 => 128, relu, pad=1),  [90m# 73_856 parameters[39m
            MaxPool((2, 2)),
          ),
        ),
        VGGBlock(
          Chain(
            Conv((3, 3), 128 => 256, relu, pad=1),  [90m# 295_168 parameters[39m
            Conv((3, 3), 256 => 256, relu, pad=1),  [90m# 590_080 parameters[39m
            MaxPool((2, 2)),
          ),
        ),
        VGGBlock(
          Chain(
            Conv((3, 3), 256 => 512, relu, pad=1),  [90m# 1_180_160 parameters[39m
            Conv((3, 3), 512 => 512, relu, pad=1),  [90m# 2_359_808 parameters[39m
            MaxPool((2, 2)),
          ),
        ),
        VGGBlock(
          Chain(
            Conv((3, 3), 512 => 512, relu, pad=1),  [90m# 2_359_808 parameters[39m
            Conv((3, 3), 512 => 512, relu, pad=1),  [90m# 2_359_808 parameters[39m
            MaxPool((2, 2)),
          ),
        ),
        Flux.flatten,
        Dense(25088 => 4096, relu),         [90m# 102_764_544 parameters[39m
        Dropout(0.5),
        Dense(4096 => 4096, relu),          [90m# 16_781_312 parameters[39m
        Dropout(0.5),
        Dense(4096 => 10),                  [90m# 40_970 parameters[39m
        NNlib.softmax,
      ),
    ) [90m                  # Total: 22 arrays, [39m128_806_154 parameters, 491.359 MiB.



As you can see, we halve height and width at each block, finally reaching a height and width of 7 before flattening the representations for processing by the fully connected part of the network. Simonyan and Zisserman (2014) described several other variants of VGG. In fact, it has become the norm to propose families of networks with different speed–accuracy trade-off when introducing a new architecture.


## Training

**Since VGG-11 is computationally more demanding than AlexNet
we construct a network with a smaller number of channels.**
This is more than sufficient for training on Fashion-MNIST.
The **model training** process is similar to that of AlexNet in :numref:`sec_alexnet`. 
Again observe the close match between validation and training loss, 
suggesting only a small amount of overfitting.



```julia
model = VGGNet(((1, 16), (1, 32), (2, 64), (2, 128), (2, 128)))
data = d2lai.FashionMNISTData(batchsize = 128, resize = (224, 224))
opt = Descent(0.01)
trainer = Trainer(model, data, opt; max_epochs = 10, gpu = true, board_yscale = :identity)
d2lai.fit(trainer);
```

    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mTrain Loss: 0.48188314, Val Loss: 0.3698047, Val Acc: 0.875
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mTrain Loss: 0.5084389, Val Loss: 0.25148895, Val Acc: 0.875
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mTrain Loss: 0.2939905, Val Loss: 0.26616976, Val Acc: 0.875
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mTrain Loss: 0.3377501, Val Loss: 0.21927431, Val Acc: 0.875
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mTrain Loss: 0.22468491, Val Loss: 0.18230689, Val Acc: 0.9375
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mTrain Loss: 0.34149575, Val Loss: 0.17228371, Val Acc: 0.9375
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mTrain Loss: 0.32494318, Val Loss: 0.15211411, Val Acc: 0.9375
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mTrain Loss: 0.33381274, Val Loss: 0.15485266, Val Acc: 0.9375
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mTrain Loss: 0.27774346, Val Loss: 0.13706203, Val Acc: 0.875
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mTrain Loss: 0.31419504, Val Loss: 0.14435501, Val Acc: 1.0



<?xml version="1.0" encoding="utf-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="600" height="400" viewBox="0 0 2400 1600">
<defs>
  <clipPath id="clip420">
    <rect x="0" y="0" width="2400" height="1600"/>
  </clipPath>
</defs>
<path clip-path="url(#clip420)" d="M0 1600 L2400 1600 L2400 0 L0 0  Z" fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<defs>
  <clipPath id="clip421">
    <rect x="480" y="0" width="1681" height="1600"/>
  </clipPath>
</defs>
<path clip-path="url(#clip420)" d="M156.598 1423.18 L2352.76 1423.18 L2352.76 47.2441 L156.598 47.2441  Z" fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<defs>
  <clipPath id="clip422">
    <rect x="156" y="47" width="2197" height="1377"/>
  </clipPath>
</defs>
<polyline clip-path="url(#clip422)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="448.959,1423.18 448.959,47.2441 "/>
<polyline clip-path="url(#clip422)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="909.369,1423.18 909.369,47.2441 "/>
<polyline clip-path="url(#clip422)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="1369.78,1423.18 1369.78,47.2441 "/>
<polyline clip-path="url(#clip422)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="1830.19,1423.18 1830.19,47.2441 "/>
<polyline clip-path="url(#clip422)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="2290.6,1423.18 2290.6,47.2441 "/>
<polyline clip-path="url(#clip422)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="156.598,1405.05 2352.76,1405.05 "/>
<polyline clip-path="url(#clip422)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="156.598,1200.1 2352.76,1200.1 "/>
<polyline clip-path="url(#clip422)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="156.598,995.151 2352.76,995.151 "/>
<polyline clip-path="url(#clip422)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="156.598,790.201 2352.76,790.201 "/>
<polyline clip-path="url(#clip422)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="156.598,585.25 2352.76,585.25 "/>
<polyline clip-path="url(#clip422)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="156.598,380.3 2352.76,380.3 "/>
<polyline clip-path="url(#clip422)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="156.598,175.349 2352.76,175.349 "/>
<polyline clip-path="url(#clip420)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="156.598,1423.18 2352.76,1423.18 "/>
<polyline clip-path="url(#clip420)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="448.959,1423.18 448.959,1404.28 "/>
<polyline clip-path="url(#clip420)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="909.369,1423.18 909.369,1404.28 "/>
<polyline clip-path="url(#clip420)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1369.78,1423.18 1369.78,1404.28 "/>
<polyline clip-path="url(#clip420)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1830.19,1423.18 1830.19,1404.28 "/>
<polyline clip-path="url(#clip420)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="2290.6,1423.18 2290.6,1404.28 "/>
<path clip-path="url(#clip420)" d="M443.612 1481.64 L459.931 1481.64 L459.931 1485.58 L437.987 1485.58 L437.987 1481.64 Q440.649 1478.89 445.232 1474.26 Q449.838 1469.61 451.019 1468.27 Q453.264 1465.74 454.144 1464.01 Q455.047 1462.25 455.047 1460.56 Q455.047 1457.8 453.102 1456.07 Q451.181 1454.33 448.079 1454.33 Q445.88 1454.33 443.426 1455.09 Q440.996 1455.86 438.218 1457.41 L438.218 1452.69 Q441.042 1451.55 443.496 1450.97 Q445.95 1450.39 447.987 1450.39 Q453.357 1450.39 456.551 1453.08 Q459.746 1455.77 459.746 1460.26 Q459.746 1462.39 458.936 1464.31 Q458.149 1466.2 456.042 1468.8 Q455.463 1469.47 452.362 1472.69 Q449.26 1475.88 443.612 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M912.379 1455.09 L900.573 1473.54 L912.379 1473.54 L912.379 1455.09 M911.152 1451.02 L917.031 1451.02 L917.031 1473.54 L921.962 1473.54 L921.962 1477.43 L917.031 1477.43 L917.031 1485.58 L912.379 1485.58 L912.379 1477.43 L896.777 1477.43 L896.777 1472.92 L911.152 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M1370.18 1466.44 Q1367.04 1466.44 1365.18 1468.59 Q1363.36 1470.74 1363.36 1474.49 Q1363.36 1478.22 1365.18 1480.39 Q1367.04 1482.55 1370.18 1482.55 Q1373.33 1482.55 1375.16 1480.39 Q1377.01 1478.22 1377.01 1474.49 Q1377.01 1470.74 1375.16 1468.59 Q1373.33 1466.44 1370.18 1466.44 M1379.47 1451.78 L1379.47 1456.04 Q1377.71 1455.21 1375.9 1454.77 Q1374.12 1454.33 1372.36 1454.33 Q1367.73 1454.33 1365.28 1457.45 Q1362.85 1460.58 1362.5 1466.9 Q1363.87 1464.89 1365.93 1463.82 Q1367.99 1462.73 1370.46 1462.73 Q1375.67 1462.73 1378.68 1465.9 Q1381.71 1469.05 1381.71 1474.49 Q1381.71 1479.82 1378.56 1483.03 Q1375.42 1486.25 1370.18 1486.25 Q1364.19 1486.25 1361.02 1481.67 Q1357.85 1477.06 1357.85 1468.33 Q1357.85 1460.14 1361.74 1455.28 Q1365.62 1450.39 1372.18 1450.39 Q1373.93 1450.39 1375.72 1450.74 Q1377.52 1451.09 1379.47 1451.78 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M1830.19 1469.17 Q1826.86 1469.17 1824.94 1470.95 Q1823.04 1472.73 1823.04 1475.86 Q1823.04 1478.98 1824.94 1480.77 Q1826.86 1482.55 1830.19 1482.55 Q1833.52 1482.55 1835.44 1480.77 Q1837.37 1478.96 1837.37 1475.86 Q1837.37 1472.73 1835.44 1470.95 Q1833.55 1469.17 1830.19 1469.17 M1825.51 1467.18 Q1822.5 1466.44 1820.82 1464.38 Q1819.15 1462.32 1819.15 1459.35 Q1819.15 1455.21 1822.09 1452.8 Q1825.05 1450.39 1830.19 1450.39 Q1835.35 1450.39 1838.29 1452.8 Q1841.23 1455.21 1841.23 1459.35 Q1841.23 1462.32 1839.54 1464.38 Q1837.88 1466.44 1834.89 1467.18 Q1838.27 1467.96 1840.14 1470.26 Q1842.04 1472.55 1842.04 1475.86 Q1842.04 1480.88 1838.96 1483.57 Q1835.91 1486.25 1830.19 1486.25 Q1824.47 1486.25 1821.39 1483.57 Q1818.34 1480.88 1818.34 1475.86 Q1818.34 1472.55 1820.24 1470.26 Q1822.13 1467.96 1825.51 1467.18 M1823.8 1459.79 Q1823.8 1462.48 1825.47 1463.98 Q1827.16 1465.49 1830.19 1465.49 Q1833.2 1465.49 1834.89 1463.98 Q1836.6 1462.48 1836.6 1459.79 Q1836.6 1457.11 1834.89 1455.6 Q1833.2 1454.1 1830.19 1454.1 Q1827.16 1454.1 1825.47 1455.6 Q1823.8 1457.11 1823.8 1459.79 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M2265.29 1481.64 L2272.93 1481.64 L2272.93 1455.28 L2264.62 1456.95 L2264.62 1452.69 L2272.88 1451.02 L2277.56 1451.02 L2277.56 1481.64 L2285.2 1481.64 L2285.2 1485.58 L2265.29 1485.58 L2265.29 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M2304.64 1454.1 Q2301.03 1454.1 2299.2 1457.66 Q2297.39 1461.2 2297.39 1468.33 Q2297.39 1475.44 2299.2 1479.01 Q2301.03 1482.55 2304.64 1482.55 Q2308.27 1482.55 2310.08 1479.01 Q2311.91 1475.44 2311.91 1468.33 Q2311.91 1461.2 2310.08 1457.66 Q2308.27 1454.1 2304.64 1454.1 M2304.64 1450.39 Q2310.45 1450.39 2313.51 1455 Q2316.58 1459.58 2316.58 1468.33 Q2316.58 1477.06 2313.51 1481.67 Q2310.45 1486.25 2304.64 1486.25 Q2298.83 1486.25 2295.75 1481.67 Q2292.7 1477.06 2292.7 1468.33 Q2292.7 1459.58 2295.75 1455 Q2298.83 1450.39 2304.64 1450.39 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M1174.87 1548.76 L1174.87 1551.62 L1147.94 1551.62 Q1148.32 1557.67 1151.57 1560.85 Q1154.85 1564 1160.67 1564 Q1164.05 1564 1167.2 1563.17 Q1170.38 1562.35 1173.5 1560.69 L1173.5 1566.23 Q1170.35 1567.57 1167.04 1568.27 Q1163.73 1568.97 1160.32 1568.97 Q1151.79 1568.97 1146.79 1564 Q1141.83 1559.04 1141.83 1550.57 Q1141.83 1541.82 1146.54 1536.69 Q1151.28 1531.54 1159.3 1531.54 Q1166.5 1531.54 1170.67 1536.18 Q1174.87 1540.8 1174.87 1548.76 M1169.01 1547.04 Q1168.95 1542.23 1166.31 1539.37 Q1163.7 1536.5 1159.37 1536.5 Q1154.46 1536.5 1151.5 1539.27 Q1148.58 1542.04 1148.13 1547.07 L1169.01 1547.04 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M1190.14 1562.7 L1190.14 1581.6 L1184.26 1581.6 L1184.26 1532.4 L1190.14 1532.4 L1190.14 1537.81 Q1191.99 1534.62 1194.79 1533.1 Q1197.62 1531.54 1201.54 1531.54 Q1208.03 1531.54 1212.07 1536.69 Q1216.15 1541.85 1216.15 1550.25 Q1216.15 1558.65 1212.07 1563.81 Q1208.03 1568.97 1201.54 1568.97 Q1197.62 1568.97 1194.79 1567.44 Q1191.99 1565.88 1190.14 1562.7 M1210.07 1550.25 Q1210.07 1543.79 1207.4 1540.13 Q1204.75 1536.44 1200.11 1536.44 Q1195.46 1536.44 1192.79 1540.13 Q1190.14 1543.79 1190.14 1550.25 Q1190.14 1556.71 1192.79 1560.4 Q1195.46 1564.07 1200.11 1564.07 Q1204.75 1564.07 1207.4 1560.4 Q1210.07 1556.71 1210.07 1550.25 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M1239.67 1536.5 Q1234.96 1536.5 1232.22 1540.19 Q1229.48 1543.85 1229.48 1550.25 Q1229.48 1556.65 1232.19 1560.34 Q1234.93 1564 1239.67 1564 Q1244.35 1564 1247.09 1560.31 Q1249.82 1556.62 1249.82 1550.25 Q1249.82 1543.92 1247.09 1540.23 Q1244.35 1536.5 1239.67 1536.5 M1239.67 1531.54 Q1247.31 1531.54 1251.67 1536.5 Q1256.03 1541.47 1256.03 1550.25 Q1256.03 1559 1251.67 1564 Q1247.31 1568.97 1239.67 1568.97 Q1232 1568.97 1227.64 1564 Q1223.31 1559 1223.31 1550.25 Q1223.31 1541.47 1227.64 1536.5 Q1232 1531.54 1239.67 1531.54 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M1291.39 1533.76 L1291.39 1539.24 Q1288.91 1537.87 1286.39 1537.2 Q1283.91 1536.5 1281.37 1536.5 Q1275.67 1536.5 1272.52 1540.13 Q1269.37 1543.73 1269.37 1550.25 Q1269.37 1556.78 1272.52 1560.4 Q1275.67 1564 1281.37 1564 Q1283.91 1564 1286.39 1563.33 Q1288.91 1562.63 1291.39 1561.26 L1291.39 1566.68 Q1288.94 1567.82 1286.3 1568.39 Q1283.69 1568.97 1280.73 1568.97 Q1272.68 1568.97 1267.93 1563.91 Q1263.19 1558.85 1263.19 1550.25 Q1263.19 1541.53 1267.97 1536.53 Q1272.77 1531.54 1281.11 1531.54 Q1283.82 1531.54 1286.39 1532.11 Q1288.97 1532.65 1291.39 1533.76 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M1331.21 1546.53 L1331.21 1568.04 L1325.35 1568.04 L1325.35 1546.72 Q1325.35 1541.66 1323.38 1539.14 Q1321.41 1536.63 1317.46 1536.63 Q1312.72 1536.63 1309.98 1539.65 Q1307.24 1542.68 1307.24 1547.9 L1307.24 1568.04 L1301.35 1568.04 L1301.35 1518.52 L1307.24 1518.52 L1307.24 1537.93 Q1309.34 1534.72 1312.18 1533.13 Q1315.04 1531.54 1318.76 1531.54 Q1324.91 1531.54 1328.06 1535.36 Q1331.21 1539.14 1331.21 1546.53 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M1365.62 1533.45 L1365.62 1538.98 Q1363.13 1537.71 1360.46 1537.07 Q1357.79 1536.44 1354.92 1536.44 Q1350.56 1536.44 1348.36 1537.77 Q1346.2 1539.11 1346.2 1541.79 Q1346.2 1543.82 1347.76 1545 Q1349.32 1546.15 1354.03 1547.2 L1356.04 1547.64 Q1362.27 1548.98 1364.88 1551.43 Q1367.53 1553.85 1367.53 1558.21 Q1367.53 1563.17 1363.58 1566.07 Q1359.66 1568.97 1352.79 1568.97 Q1349.92 1568.97 1346.8 1568.39 Q1343.72 1567.85 1340.28 1566.74 L1340.28 1560.69 Q1343.53 1562.38 1346.68 1563.24 Q1349.83 1564.07 1352.92 1564.07 Q1357.05 1564.07 1359.28 1562.66 Q1361.51 1561.23 1361.51 1558.65 Q1361.51 1556.27 1359.89 1554.99 Q1358.29 1553.72 1352.85 1552.54 L1350.82 1552.07 Q1345.37 1550.92 1342.95 1548.56 Q1340.53 1546.18 1340.53 1542.04 Q1340.53 1537.01 1344.1 1534.27 Q1347.66 1531.54 1354.22 1531.54 Q1357.47 1531.54 1360.33 1532.01 Q1363.2 1532.49 1365.62 1533.45 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><polyline clip-path="url(#clip420)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="156.598,1423.18 156.598,47.2441 "/>
<polyline clip-path="url(#clip420)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="156.598,1405.05 175.496,1405.05 "/>
<polyline clip-path="url(#clip420)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="156.598,1200.1 175.496,1200.1 "/>
<polyline clip-path="url(#clip420)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="156.598,995.151 175.496,995.151 "/>
<polyline clip-path="url(#clip420)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="156.598,790.201 175.496,790.201 "/>
<polyline clip-path="url(#clip420)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="156.598,585.25 175.496,585.25 "/>
<polyline clip-path="url(#clip420)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="156.598,380.3 175.496,380.3 "/>
<polyline clip-path="url(#clip420)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="156.598,175.349 175.496,175.349 "/>
<path clip-path="url(#clip420)" d="M65.0198 1390.85 Q61.4087 1390.85 59.58 1394.42 Q57.7745 1397.96 57.7745 1405.09 Q57.7745 1412.19 59.58 1415.76 Q61.4087 1419.3 65.0198 1419.3 Q68.6541 1419.3 70.4596 1415.76 Q72.2883 1412.19 72.2883 1405.09 Q72.2883 1397.96 70.4596 1394.42 Q68.6541 1390.85 65.0198 1390.85 M65.0198 1387.15 Q70.83 1387.15 73.8855 1391.75 Q76.9642 1396.34 76.9642 1405.09 Q76.9642 1413.81 73.8855 1418.42 Q70.83 1423 65.0198 1423 Q59.2097 1423 56.131 1418.42 Q53.0754 1413.81 53.0754 1405.09 Q53.0754 1396.34 56.131 1391.75 Q59.2097 1387.15 65.0198 1387.15 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M85.1818 1416.45 L90.066 1416.45 L90.066 1422.33 L85.1818 1422.33 L85.1818 1416.45 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M104.279 1418.4 L120.598 1418.4 L120.598 1422.33 L98.6539 1422.33 L98.6539 1418.4 Q101.316 1415.64 105.899 1411.01 Q110.506 1406.36 111.686 1405.02 Q113.932 1402.49 114.811 1400.76 Q115.714 1399 115.714 1397.31 Q115.714 1394.55 113.77 1392.82 Q111.848 1391.08 108.746 1391.08 Q106.547 1391.08 104.094 1391.85 Q101.663 1392.61 98.8854 1394.16 L98.8854 1389.44 Q101.709 1388.3 104.163 1387.73 Q106.617 1387.15 108.654 1387.15 Q114.024 1387.15 117.219 1389.83 Q120.413 1392.52 120.413 1397.01 Q120.413 1399.14 119.603 1401.06 Q118.816 1402.96 116.709 1405.55 Q116.131 1406.22 113.029 1409.44 Q109.927 1412.63 104.279 1418.4 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M62.9365 1185.9 Q59.3254 1185.9 57.4967 1189.47 Q55.6912 1193.01 55.6912 1200.14 Q55.6912 1207.24 57.4967 1210.81 Q59.3254 1214.35 62.9365 1214.35 Q66.5707 1214.35 68.3763 1210.81 Q70.205 1207.24 70.205 1200.14 Q70.205 1193.01 68.3763 1189.47 Q66.5707 1185.9 62.9365 1185.9 M62.9365 1182.2 Q68.7467 1182.2 71.8022 1186.8 Q74.8809 1191.39 74.8809 1200.14 Q74.8809 1208.86 71.8022 1213.47 Q68.7467 1218.05 62.9365 1218.05 Q57.1264 1218.05 54.0477 1213.47 Q50.9921 1208.86 50.9921 1200.14 Q50.9921 1191.39 54.0477 1186.8 Q57.1264 1182.2 62.9365 1182.2 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M83.0984 1211.5 L87.9827 1211.5 L87.9827 1217.38 L83.0984 1217.38 L83.0984 1211.5 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M111.015 1186.9 L99.2095 1205.35 L111.015 1205.35 L111.015 1186.9 M109.788 1182.82 L115.668 1182.82 L115.668 1205.35 L120.598 1205.35 L120.598 1209.23 L115.668 1209.23 L115.668 1217.38 L111.015 1217.38 L111.015 1209.23 L95.4132 1209.23 L95.4132 1204.72 L109.788 1182.82 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M63.2606 980.95 Q59.6495 980.95 57.8208 984.515 Q56.0152 988.057 56.0152 995.186 Q56.0152 1002.29 57.8208 1005.86 Q59.6495 1009.4 63.2606 1009.4 Q66.8948 1009.4 68.7004 1005.86 Q70.5291 1002.29 70.5291 995.186 Q70.5291 988.057 68.7004 984.515 Q66.8948 980.95 63.2606 980.95 M63.2606 977.246 Q69.0707 977.246 72.1263 981.853 Q75.205 986.436 75.205 995.186 Q75.205 1003.91 72.1263 1008.52 Q69.0707 1013.1 63.2606 1013.1 Q57.4504 1013.1 54.3717 1008.52 Q51.3162 1003.91 51.3162 995.186 Q51.3162 986.436 54.3717 981.853 Q57.4504 977.246 63.2606 977.246 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M83.4225 1006.55 L88.3067 1006.55 L88.3067 1012.43 L83.4225 1012.43 L83.4225 1006.55 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M109.071 993.288 Q105.922 993.288 104.071 995.441 Q102.242 997.594 102.242 1001.34 Q102.242 1005.07 104.071 1007.25 Q105.922 1009.4 109.071 1009.4 Q112.219 1009.4 114.047 1007.25 Q115.899 1005.07 115.899 1001.34 Q115.899 997.594 114.047 995.441 Q112.219 993.288 109.071 993.288 M118.353 978.635 L118.353 982.895 Q116.594 982.061 114.788 981.621 Q113.006 981.182 111.246 981.182 Q106.617 981.182 104.163 984.307 Q101.733 987.432 101.385 993.751 Q102.751 991.737 104.811 990.672 Q106.871 989.584 109.348 989.584 Q114.557 989.584 117.566 992.756 Q120.598 995.904 120.598 1001.34 Q120.598 1006.67 117.45 1009.89 Q114.302 1013.1 109.071 1013.1 Q103.075 1013.1 99.9039 1008.52 Q96.7326 1003.91 96.7326 995.186 Q96.7326 986.992 100.621 982.131 Q104.51 977.246 111.061 977.246 Q112.82 977.246 114.603 977.594 Q116.408 977.941 118.353 978.635 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M63.5152 776 Q59.9041 776 58.0754 779.564 Q56.2699 783.106 56.2699 790.236 Q56.2699 797.342 58.0754 800.907 Q59.9041 804.449 63.5152 804.449 Q67.1494 804.449 68.955 800.907 Q70.7837 797.342 70.7837 790.236 Q70.7837 783.106 68.955 779.564 Q67.1494 776 63.5152 776 M63.5152 772.296 Q69.3254 772.296 72.3809 776.902 Q75.4596 781.486 75.4596 790.236 Q75.4596 798.962 72.3809 803.569 Q69.3254 808.152 63.5152 808.152 Q57.7051 808.152 54.6264 803.569 Q51.5708 798.962 51.5708 790.236 Q51.5708 781.486 54.6264 776.902 Q57.7051 772.296 63.5152 772.296 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M83.6771 801.601 L88.5614 801.601 L88.5614 807.481 L83.6771 807.481 L83.6771 801.601 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M108.746 791.069 Q105.413 791.069 103.492 792.851 Q101.594 794.634 101.594 797.759 Q101.594 800.884 103.492 802.666 Q105.413 804.449 108.746 804.449 Q112.08 804.449 114.001 802.666 Q115.922 800.861 115.922 797.759 Q115.922 794.634 114.001 792.851 Q112.103 791.069 108.746 791.069 M104.071 789.078 Q101.061 788.337 99.3715 786.277 Q97.7048 784.217 97.7048 781.254 Q97.7048 777.111 100.645 774.703 Q103.608 772.296 108.746 772.296 Q113.908 772.296 116.848 774.703 Q119.788 777.111 119.788 781.254 Q119.788 784.217 118.098 786.277 Q116.432 788.337 113.445 789.078 Q116.825 789.865 118.7 792.157 Q120.598 794.449 120.598 797.759 Q120.598 802.782 117.52 805.467 Q114.464 808.152 108.746 808.152 Q103.029 808.152 99.9502 805.467 Q96.8947 802.782 96.8947 797.759 Q96.8947 794.449 98.7928 792.157 Q100.691 789.865 104.071 789.078 M102.358 781.694 Q102.358 784.379 104.024 785.884 Q105.714 787.388 108.746 787.388 Q111.756 787.388 113.445 785.884 Q115.158 784.379 115.158 781.694 Q115.158 779.009 113.445 777.504 Q111.756 776 108.746 776 Q105.714 776 104.024 777.504 Q102.358 779.009 102.358 781.694 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M54.2328 598.595 L61.8717 598.595 L61.8717 572.23 L53.5616 573.896 L53.5616 569.637 L61.8254 567.97 L66.5013 567.97 L66.5013 598.595 L74.1402 598.595 L74.1402 602.53 L54.2328 602.53 L54.2328 598.595 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M83.5845 596.651 L88.4688 596.651 L88.4688 602.53 L83.5845 602.53 L83.5845 596.651 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M108.654 571.049 Q105.043 571.049 103.214 574.614 Q101.409 578.156 101.409 585.285 Q101.409 592.392 103.214 595.956 Q105.043 599.498 108.654 599.498 Q112.288 599.498 114.094 595.956 Q115.922 592.392 115.922 585.285 Q115.922 578.156 114.094 574.614 Q112.288 571.049 108.654 571.049 M108.654 567.345 Q114.464 567.345 117.52 571.952 Q120.598 576.535 120.598 585.285 Q120.598 594.012 117.52 598.618 Q114.464 603.202 108.654 603.202 Q102.844 603.202 99.765 598.618 Q96.7095 594.012 96.7095 585.285 Q96.7095 576.535 99.765 571.952 Q102.844 567.345 108.654 567.345 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M55.8301 393.645 L63.4689 393.645 L63.4689 367.279 L55.1588 368.946 L55.1588 364.686 L63.4226 363.02 L68.0985 363.02 L68.0985 393.645 L75.7374 393.645 L75.7374 397.58 L55.8301 397.58 L55.8301 393.645 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M85.1818 391.7 L90.066 391.7 L90.066 397.58 L85.1818 397.58 L85.1818 391.7 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M104.279 393.645 L120.598 393.645 L120.598 397.58 L98.6539 397.58 L98.6539 393.645 Q101.316 390.89 105.899 386.26 Q110.506 381.608 111.686 380.265 Q113.932 377.742 114.811 376.006 Q115.714 374.247 115.714 372.557 Q115.714 369.802 113.77 368.066 Q111.848 366.33 108.746 366.33 Q106.547 366.33 104.094 367.094 Q101.663 367.858 98.8854 369.409 L98.8854 364.686 Q101.709 363.552 104.163 362.974 Q106.617 362.395 108.654 362.395 Q114.024 362.395 117.219 365.08 Q120.413 367.765 120.413 372.256 Q120.413 374.386 119.603 376.307 Q118.816 378.205 116.709 380.798 Q116.131 381.469 113.029 384.686 Q109.927 387.881 104.279 393.645 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M53.7467 188.694 L61.3856 188.694 L61.3856 162.329 L53.0754 163.995 L53.0754 159.736 L61.3393 158.069 L66.0152 158.069 L66.0152 188.694 L73.654 188.694 L73.654 192.629 L53.7467 192.629 L53.7467 188.694 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M83.0984 186.75 L87.9827 186.75 L87.9827 192.629 L83.0984 192.629 L83.0984 186.75 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M111.015 162.143 L99.2095 180.592 L111.015 180.592 L111.015 162.143 M109.788 158.069 L115.668 158.069 L115.668 180.592 L120.598 180.592 L120.598 184.481 L115.668 184.481 L115.668 192.629 L111.015 192.629 L111.015 184.481 L95.4132 184.481 L95.4132 179.967 L109.788 158.069 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><polyline clip-path="url(#clip422)" style="stroke:#009af9; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="218.754,86.1857 448.959,1108.87 679.164,1223.37 909.369,1269.73 1139.57,1299.29 1369.78,1322.23 1599.98,1341.2 1830.19,1357.42 2060.4,1371.89 2290.6,1384.24 "/>
<polyline clip-path="url(#clip422)" style="stroke:#e26f46; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="218.754,907.338 448.959,1113.18 679.164,1176.26 909.369,1195.86 1139.57,1238.73 1369.78,1260.06 1599.98,1274.85 1830.19,1281.07 2060.4,1297.47 2290.6,1307.4 "/>
<polyline clip-path="url(#clip422)" style="stroke:#3da44d; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="218.754,848.33 448.959,764.116 679.164,741.618 909.369,735.335 1139.57,719.932 1369.78,712.838 1599.98,706.96 1830.19,704.832 2060.4,698.853 2290.6,695.103 "/>
<path clip-path="url(#clip420)" d="M1837.9 300.469 L2279.55 300.469 L2279.55 93.1086 L1837.9 93.1086  Z" fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<polyline clip-path="url(#clip420)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1837.9,300.469 2279.55,300.469 2279.55,93.1086 1837.9,93.1086 1837.9,300.469 "/>
<polyline clip-path="url(#clip420)" style="stroke:#009af9; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1862.3,144.949 2008.71,144.949 "/>
<path clip-path="url(#clip420)" d="M2040.52 128.942 L2040.52 136.303 L2049.29 136.303 L2049.29 139.613 L2040.52 139.613 L2040.52 153.687 Q2040.52 156.858 2041.38 157.761 Q2042.26 158.664 2044.92 158.664 L2049.29 158.664 L2049.29 162.229 L2044.92 162.229 Q2039.99 162.229 2038.11 160.4 Q2036.24 158.548 2036.24 153.687 L2036.24 139.613 L2033.11 139.613 L2033.11 136.303 L2036.24 136.303 L2036.24 128.942 L2040.52 128.942 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M2069.92 140.284 Q2069.2 139.868 2068.34 139.682 Q2067.51 139.474 2066.49 139.474 Q2062.88 139.474 2060.94 141.835 Q2059.02 144.173 2059.02 148.571 L2059.02 162.229 L2054.73 162.229 L2054.73 136.303 L2059.02 136.303 L2059.02 140.331 Q2060.36 137.969 2062.51 136.835 Q2064.66 135.678 2067.74 135.678 Q2068.18 135.678 2068.71 135.747 Q2069.25 135.794 2069.9 135.909 L2069.92 140.284 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M2086.17 149.196 Q2081.01 149.196 2079.02 150.377 Q2077.02 151.557 2077.02 154.405 Q2077.02 156.673 2078.51 158.016 Q2080.01 159.335 2082.58 159.335 Q2086.12 159.335 2088.25 156.835 Q2090.4 154.312 2090.4 150.145 L2090.4 149.196 L2086.17 149.196 M2094.66 147.437 L2094.66 162.229 L2090.4 162.229 L2090.4 158.293 Q2088.95 160.655 2086.77 161.789 Q2084.59 162.9 2081.45 162.9 Q2077.46 162.9 2075.1 160.678 Q2072.77 158.432 2072.77 154.682 Q2072.77 150.307 2075.68 148.085 Q2078.62 145.863 2084.43 145.863 L2090.4 145.863 L2090.4 145.446 Q2090.4 142.507 2088.46 140.909 Q2086.54 139.289 2083.04 139.289 Q2080.82 139.289 2078.71 139.821 Q2076.61 140.354 2074.66 141.419 L2074.66 137.483 Q2077 136.581 2079.2 136.141 Q2081.4 135.678 2083.48 135.678 Q2089.11 135.678 2091.89 138.594 Q2094.66 141.511 2094.66 147.437 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M2103.44 136.303 L2107.7 136.303 L2107.7 162.229 L2103.44 162.229 L2103.44 136.303 M2103.44 126.21 L2107.7 126.21 L2107.7 131.604 L2103.44 131.604 L2103.44 126.21 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M2138.16 146.581 L2138.16 162.229 L2133.9 162.229 L2133.9 146.719 Q2133.9 143.039 2132.46 141.21 Q2131.03 139.382 2128.16 139.382 Q2124.71 139.382 2122.72 141.581 Q2120.73 143.78 2120.73 147.576 L2120.73 162.229 L2116.45 162.229 L2116.45 136.303 L2120.73 136.303 L2120.73 140.331 Q2122.26 137.993 2124.32 136.835 Q2126.4 135.678 2129.11 135.678 Q2133.58 135.678 2135.87 138.456 Q2138.16 141.21 2138.16 146.581 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M2166.35 170.099 L2166.35 173.409 L2141.72 173.409 L2141.72 170.099 L2166.35 170.099 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M2170.36 126.21 L2174.62 126.21 L2174.62 162.229 L2170.36 162.229 L2170.36 126.21 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M2193.58 139.289 Q2190.15 139.289 2188.16 141.974 Q2186.17 144.636 2186.17 149.289 Q2186.17 153.942 2188.14 156.627 Q2190.13 159.289 2193.58 159.289 Q2196.98 159.289 2198.97 156.604 Q2200.96 153.918 2200.96 149.289 Q2200.96 144.682 2198.97 141.997 Q2196.98 139.289 2193.58 139.289 M2193.58 135.678 Q2199.13 135.678 2202.3 139.289 Q2205.47 142.9 2205.47 149.289 Q2205.47 155.655 2202.3 159.289 Q2199.13 162.9 2193.58 162.9 Q2188 162.9 2184.83 159.289 Q2181.68 155.655 2181.68 149.289 Q2181.68 142.9 2184.83 139.289 Q2188 135.678 2193.58 135.678 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M2229.06 137.067 L2229.06 141.094 Q2227.26 140.169 2225.31 139.706 Q2223.37 139.243 2221.28 139.243 Q2218.11 139.243 2216.51 140.215 Q2214.94 141.187 2214.94 143.131 Q2214.94 144.613 2216.08 145.469 Q2217.21 146.303 2220.64 147.067 L2222.09 147.391 Q2226.63 148.363 2228.53 150.145 Q2230.45 151.905 2230.45 155.076 Q2230.45 158.687 2227.58 160.793 Q2224.73 162.9 2219.73 162.9 Q2217.65 162.9 2215.38 162.483 Q2213.14 162.09 2210.64 161.28 L2210.64 156.881 Q2213 158.108 2215.29 158.733 Q2217.58 159.335 2219.83 159.335 Q2222.83 159.335 2224.45 158.317 Q2226.07 157.275 2226.07 155.4 Q2226.07 153.664 2224.89 152.738 Q2223.74 151.812 2219.78 150.956 L2218.3 150.608 Q2214.34 149.775 2212.58 148.062 Q2210.82 146.326 2210.82 143.317 Q2210.82 139.659 2213.41 137.669 Q2216.01 135.678 2220.77 135.678 Q2223.14 135.678 2225.22 136.025 Q2227.3 136.372 2229.06 137.067 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M2253.76 137.067 L2253.76 141.094 Q2251.95 140.169 2250.01 139.706 Q2248.07 139.243 2245.98 139.243 Q2242.81 139.243 2241.21 140.215 Q2239.64 141.187 2239.64 143.131 Q2239.64 144.613 2240.77 145.469 Q2241.91 146.303 2245.33 147.067 L2246.79 147.391 Q2251.33 148.363 2253.23 150.145 Q2255.15 151.905 2255.15 155.076 Q2255.15 158.687 2252.28 160.793 Q2249.43 162.9 2244.43 162.9 Q2242.35 162.9 2240.08 162.483 Q2237.83 162.09 2235.33 161.28 L2235.33 156.881 Q2237.7 158.108 2239.99 158.733 Q2242.28 159.335 2244.52 159.335 Q2247.53 159.335 2249.15 158.317 Q2250.77 157.275 2250.77 155.4 Q2250.77 153.664 2249.59 152.738 Q2248.44 151.812 2244.48 150.956 L2243 150.608 Q2239.04 149.775 2237.28 148.062 Q2235.52 146.326 2235.52 143.317 Q2235.52 139.659 2238.11 137.669 Q2240.7 135.678 2245.47 135.678 Q2247.83 135.678 2249.92 136.025 Q2252 136.372 2253.76 137.067 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><polyline clip-path="url(#clip420)" style="stroke:#e26f46; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1862.3,196.789 2008.71,196.789 "/>
<path clip-path="url(#clip420)" d="M2033.11 188.143 L2037.63 188.143 L2045.73 209.902 L2053.83 188.143 L2058.34 188.143 L2048.62 214.069 L2042.84 214.069 L2033.11 188.143 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M2076.01 201.036 Q2070.84 201.036 2068.85 202.217 Q2066.86 203.397 2066.86 206.245 Q2066.86 208.513 2068.34 209.856 Q2069.85 211.175 2072.42 211.175 Q2075.96 211.175 2078.09 208.675 Q2080.24 206.152 2080.24 201.985 L2080.24 201.036 L2076.01 201.036 M2084.5 199.277 L2084.5 214.069 L2080.24 214.069 L2080.24 210.133 Q2078.78 212.495 2076.61 213.629 Q2074.43 214.74 2071.28 214.74 Q2067.3 214.74 2064.94 212.518 Q2062.6 210.272 2062.6 206.522 Q2062.6 202.147 2065.52 199.925 Q2068.46 197.703 2074.27 197.703 L2080.24 197.703 L2080.24 197.286 Q2080.24 194.347 2078.3 192.749 Q2076.38 191.129 2072.88 191.129 Q2070.66 191.129 2068.55 191.661 Q2066.45 192.194 2064.5 193.259 L2064.5 189.323 Q2066.84 188.421 2069.04 187.981 Q2071.24 187.518 2073.32 187.518 Q2078.95 187.518 2081.72 190.434 Q2084.5 193.351 2084.5 199.277 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M2093.27 178.05 L2097.53 178.05 L2097.53 214.069 L2093.27 214.069 L2093.27 178.05 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M2126.14 221.939 L2126.14 225.249 L2101.52 225.249 L2101.52 221.939 L2126.14 221.939 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M2130.15 178.05 L2134.41 178.05 L2134.41 214.069 L2130.15 214.069 L2130.15 178.05 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M2153.37 191.129 Q2149.94 191.129 2147.95 193.814 Q2145.96 196.476 2145.96 201.129 Q2145.96 205.782 2147.93 208.467 Q2149.92 211.129 2153.37 211.129 Q2156.77 211.129 2158.76 208.444 Q2160.75 205.758 2160.75 201.129 Q2160.75 196.522 2158.76 193.837 Q2156.77 191.129 2153.37 191.129 M2153.37 187.518 Q2158.92 187.518 2162.09 191.129 Q2165.27 194.74 2165.27 201.129 Q2165.27 207.495 2162.09 211.129 Q2158.92 214.74 2153.37 214.74 Q2147.79 214.74 2144.62 211.129 Q2141.47 207.495 2141.47 201.129 Q2141.47 194.74 2144.62 191.129 Q2147.79 187.518 2153.37 187.518 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M2188.85 188.907 L2188.85 192.934 Q2187.05 192.009 2185.1 191.546 Q2183.16 191.083 2181.08 191.083 Q2177.9 191.083 2176.31 192.055 Q2174.73 193.027 2174.73 194.971 Q2174.73 196.453 2175.87 197.309 Q2177 198.143 2180.43 198.907 L2181.89 199.231 Q2186.42 200.203 2188.32 201.985 Q2190.24 203.745 2190.24 206.916 Q2190.24 210.527 2187.37 212.633 Q2184.52 214.74 2179.52 214.74 Q2177.44 214.74 2175.17 214.323 Q2172.93 213.93 2170.43 213.12 L2170.43 208.721 Q2172.79 209.948 2175.08 210.573 Q2177.37 211.175 2179.62 211.175 Q2182.63 211.175 2184.25 210.157 Q2185.87 209.115 2185.87 207.24 Q2185.87 205.504 2184.69 204.578 Q2183.53 203.652 2179.57 202.796 L2178.09 202.448 Q2174.13 201.615 2172.37 199.902 Q2170.61 198.166 2170.61 195.157 Q2170.61 191.499 2173.2 189.509 Q2175.8 187.518 2180.57 187.518 Q2182.93 187.518 2185.01 187.865 Q2187.09 188.212 2188.85 188.907 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M2213.55 188.907 L2213.55 192.934 Q2211.75 192.009 2209.8 191.546 Q2207.86 191.083 2205.77 191.083 Q2202.6 191.083 2201.01 192.055 Q2199.43 193.027 2199.43 194.971 Q2199.43 196.453 2200.57 197.309 Q2201.7 198.143 2205.13 198.907 L2206.58 199.231 Q2211.12 200.203 2213.02 201.985 Q2214.94 203.745 2214.94 206.916 Q2214.94 210.527 2212.07 212.633 Q2209.22 214.74 2204.22 214.74 Q2202.14 214.74 2199.87 214.323 Q2197.63 213.93 2195.13 213.12 L2195.13 208.721 Q2197.49 209.948 2199.78 210.573 Q2202.07 211.175 2204.32 211.175 Q2207.33 211.175 2208.95 210.157 Q2210.57 209.115 2210.57 207.24 Q2210.57 205.504 2209.39 204.578 Q2208.23 203.652 2204.27 202.796 L2202.79 202.448 Q2198.83 201.615 2197.07 199.902 Q2195.31 198.166 2195.31 195.157 Q2195.31 191.499 2197.9 189.509 Q2200.5 187.518 2205.26 187.518 Q2207.63 187.518 2209.71 187.865 Q2211.79 188.212 2213.55 188.907 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><polyline clip-path="url(#clip420)" style="stroke:#3da44d; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1862.3,248.629 2008.71,248.629 "/>
<path clip-path="url(#clip420)" d="M2033.11 239.983 L2037.63 239.983 L2045.73 261.742 L2053.83 239.983 L2058.34 239.983 L2048.62 265.909 L2042.84 265.909 L2033.11 239.983 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M2076.01 252.876 Q2070.84 252.876 2068.85 254.057 Q2066.86 255.237 2066.86 258.085 Q2066.86 260.353 2068.34 261.696 Q2069.85 263.015 2072.42 263.015 Q2075.96 263.015 2078.09 260.515 Q2080.24 257.992 2080.24 253.825 L2080.24 252.876 L2076.01 252.876 M2084.5 251.117 L2084.5 265.909 L2080.24 265.909 L2080.24 261.973 Q2078.78 264.335 2076.61 265.469 Q2074.43 266.58 2071.28 266.58 Q2067.3 266.58 2064.94 264.358 Q2062.6 262.112 2062.6 258.362 Q2062.6 253.987 2065.52 251.765 Q2068.46 249.543 2074.27 249.543 L2080.24 249.543 L2080.24 249.126 Q2080.24 246.187 2078.3 244.589 Q2076.38 242.969 2072.88 242.969 Q2070.66 242.969 2068.55 243.501 Q2066.45 244.034 2064.5 245.099 L2064.5 241.163 Q2066.84 240.261 2069.04 239.821 Q2071.24 239.358 2073.32 239.358 Q2078.95 239.358 2081.72 242.274 Q2084.5 245.191 2084.5 251.117 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M2093.27 229.89 L2097.53 229.89 L2097.53 265.909 L2093.27 265.909 L2093.27 229.89 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M2126.14 273.779 L2126.14 277.089 L2101.52 277.089 L2101.52 273.779 L2126.14 273.779 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M2141.93 252.876 Q2136.77 252.876 2134.78 254.057 Q2132.79 255.237 2132.79 258.085 Q2132.79 260.353 2134.27 261.696 Q2135.77 263.015 2138.34 263.015 Q2141.89 263.015 2144.02 260.515 Q2146.17 257.992 2146.17 253.825 L2146.17 252.876 L2141.93 252.876 M2150.43 251.117 L2150.43 265.909 L2146.17 265.909 L2146.17 261.973 Q2144.71 264.335 2142.53 265.469 Q2140.36 266.58 2137.21 266.58 Q2133.23 266.58 2130.87 264.358 Q2128.53 262.112 2128.53 258.362 Q2128.53 253.987 2131.45 251.765 Q2134.39 249.543 2140.2 249.543 L2146.17 249.543 L2146.17 249.126 Q2146.17 246.187 2144.22 244.589 Q2142.3 242.969 2138.81 242.969 Q2136.58 242.969 2134.48 243.501 Q2132.37 244.034 2130.43 245.099 L2130.43 241.163 Q2132.77 240.261 2134.96 239.821 Q2137.16 239.358 2139.25 239.358 Q2144.87 239.358 2147.65 242.274 Q2150.43 245.191 2150.43 251.117 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M2177.86 240.978 L2177.86 244.96 Q2176.05 243.964 2174.22 243.478 Q2172.42 242.969 2170.57 242.969 Q2166.42 242.969 2164.13 245.608 Q2161.84 248.224 2161.84 252.969 Q2161.84 257.714 2164.13 260.353 Q2166.42 262.969 2170.57 262.969 Q2172.42 262.969 2174.22 262.483 Q2176.05 261.973 2177.86 260.978 L2177.86 264.913 Q2176.08 265.747 2174.15 266.163 Q2172.26 266.58 2170.1 266.58 Q2164.25 266.58 2160.8 262.899 Q2157.35 259.219 2157.35 252.969 Q2157.35 246.626 2160.82 242.992 Q2164.32 239.358 2170.38 239.358 Q2172.35 239.358 2174.22 239.775 Q2176.1 240.168 2177.86 240.978 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip420)" d="M2203.92 240.978 L2203.92 244.96 Q2202.12 243.964 2200.29 243.478 Q2198.48 242.969 2196.63 242.969 Q2192.49 242.969 2190.2 245.608 Q2187.9 248.224 2187.9 252.969 Q2187.9 257.714 2190.2 260.353 Q2192.49 262.969 2196.63 262.969 Q2198.48 262.969 2200.29 262.483 Q2202.12 261.973 2203.92 260.978 L2203.92 264.913 Q2202.14 265.747 2200.22 266.163 Q2198.32 266.58 2196.17 266.58 Q2190.31 266.58 2186.86 262.899 Q2183.41 259.219 2183.41 252.969 Q2183.41 246.626 2186.89 242.992 Q2190.38 239.358 2196.45 239.358 Q2198.41 239.358 2200.29 239.775 Q2202.16 240.168 2203.92 240.978 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /></svg>






    (VGGNet{Chain{Tuple{VGGBlock{Chain{Tuple{Conv{2, 4, typeof(relu), Array{Float32, 4}, Vector{Float32}}, MaxPool{2, 4}}}}, VGGBlock{Chain{Tuple{Conv{2, 4, typeof(relu), Array{Float32, 4}, Vector{Float32}}, MaxPool{2, 4}}}}, VGGBlock{Chain{Tuple{Conv{2, 4, typeof(relu), Array{Float32, 4}, Vector{Float32}}, Conv{2, 4, typeof(relu), Array{Float32, 4}, Vector{Float32}}, MaxPool{2, 4}}}}, VGGBlock{Chain{Tuple{Conv{2, 4, typeof(relu), Array{Float32, 4}, Vector{Float32}}, Conv{2, 4, typeof(relu), Array{Float32, 4}, Vector{Float32}}, MaxPool{2, 4}}}}, VGGBlock{Chain{Tuple{Conv{2, 4, typeof(relu), Array{Float32, 4}, Vector{Float32}}, Conv{2, 4, typeof(relu), Array{Float32, 4}, Vector{Float32}}, MaxPool{2, 4}}}}, typeof(Flux.flatten), Dense{typeof(relu), Matrix{Float32}, Vector{Float32}}, Dropout{Float64, Colon, Random.TaskLocalRNG}, Dense{typeof(relu), Matrix{Float32}, Vector{Float32}}, Dropout{Float64, Colon, Random.TaskLocalRNG}, Dense{typeof(identity), Matrix{Float32}, Vector{Float32}}, typeof(softmax)}}}(Chain(VGGBlock{Chain{Tuple{Conv{2, 4, typeof(relu), Array{Float32, 4}, Vector{Float32}}, MaxPool{2, 4}}}}(Chain(Conv((3, 3), 1 => 16, relu, pad=1), MaxPool((2, 2)))), VGGBlock{Chain{Tuple{Conv{2, 4, typeof(relu), Array{Float32, 4}, Vector{Float32}}, MaxPool{2, 4}}}}(Chain(Conv((3, 3), 16 => 32, relu, pad=1), MaxPool((2, 2)))), VGGBlock{Chain{Tuple{Conv{2, 4, typeof(relu), Array{Float32, 4}, Vector{Float32}}, Conv{2, 4, typeof(relu), Array{Float32, 4}, Vector{Float32}}, MaxPool{2, 4}}}}(Chain(Conv((3, 3), 32 => 64, relu, pad=1), Conv((3, 3), 64 => 64, relu, pad=1), MaxPool((2, 2)))), VGGBlock{Chain{Tuple{Conv{2, 4, typeof(relu), Array{Float32, 4}, Vector{Float32}}, Conv{2, 4, typeof(relu), Array{Float32, 4}, Vector{Float32}}, MaxPool{2, 4}}}}(Chain(Conv((3, 3), 64 => 128, relu, pad=1), Conv((3, 3), 128 => 128, relu, pad=1), MaxPool((2, 2)))), VGGBlock{Chain{Tuple{Conv{2, 4, typeof(relu), Array{Float32, 4}, Vector{Float32}}, Conv{2, 4, typeof(relu), Array{Float32, 4}, Vector{Float32}}, MaxPool{2, 4}}}}(Chain(Conv((3, 3), 128 => 128, relu, pad=1), Conv((3, 3), 128 => 128, relu, pad=1), MaxPool((2, 2)))), flatten, Dense(6272 => 4096, relu), Dropout(0.5), Dense(4096 => 4096, relu), Dropout(0.5), Dense(4096 => 10), softmax)), (val_loss = Float32[0.30930406, 0.27966642, 0.24678184, 0.2963435, 0.3039956, 0.3340265, 0.26703936, 0.34077984, 0.2760143, 0.3184268  …  0.3769835, 0.21881735, 0.32038057, 0.33395818, 0.21082029, 0.36042413, 0.24314894, 0.18392721, 0.3268779, 0.14435501], val_acc = [0.8828125, 0.8984375, 0.8984375, 0.8984375, 0.8984375, 0.9140625, 0.9296875, 0.875, 0.90625, 0.875  …  0.8671875, 0.90625, 0.8671875, 0.875, 0.90625, 0.8828125, 0.8828125, 0.9375, 0.8671875, 1.0]))




## Summary

One might argue that VGG is the first truly modern convolutional neural network. While AlexNet introduced many of the components of what make deep learning effective at scale, it is VGG that arguably introduced key properties such as blocks of multiple convolutions and a preference for deep and narrow networks. It is also the first network that is actually an entire family of similarly parametrized models, giving the practitioner ample trade-off between complexity and speed. This is also the place where modern deep learning frameworks shine. It is no longer necessary to generate XML configuration files to specify a network but rather, to assemble said networks through simple Python code. 

More recently ParNet :cite:`Goyal.Bochkovskiy.Deng.ea.2021` demonstrated that it is possible to achieve competitive performance using a much more shallow architecture through a large number of parallel computations. This is an exciting development and there is hope that it will influence architecture designs in the future. For the remainder of the chapter, though, we will follow the path of scientific progress over the past decade. 

## Exercises


1. Compared with AlexNet, VGG is much slower in terms of computation, and it also needs more GPU memory. 
    1. Compare the number of parameters needed for AlexNet and VGG.
    1. Compare the number of floating point operations used in the convolutional layers and in the fully connected layers. 
    1. How could you reduce the computational cost created by the fully connected layers?
1. When displaying the dimensions associated with the various layers of the network, we only see the information associated with eight blocks (plus some auxiliary transforms), even though the network has 11 layers. Where did the remaining three layers go?
1. Use Table 1 in the VGG paper :cite:`Simonyan.Zisserman.2014` to construct other common models, such as VGG-16 or VGG-19.
1. Upsampling the resolution in Fashion-MNIST eight-fold from $28 \times 28$ to $224 \times 224$ dimensions is very wasteful. Try modifying the network architecture and resolution conversion, e.g., to 56 or to 84 dimensions for its input instead. Can you do so without reducing the accuracy of the network? Consult the VGG paper :cite:`Simonyan.Zisserman.2014` for ideas on adding more nonlinearities prior to downsampling.

**1**: 
    A: The parameters needed for AlexNet is 46_764_746 and for VGG-11 is 128_806_154, almost thrice to that of an AlexNet 
    C: We can use pooling to downsample more towards the end. 

**2**: Not an issue in Julia. However for Pytorch, the display shows the output of a sequential layer, i.e. collected output for all convolution layers + the pooling layer associated to a block, thats why we see less layers.

**3**: VGG 16 and VGG 19 Respectively:


```julia
vgg_16 = VGGNet(((2, 64), (2, 128), (3, 256), (3, 512), (3, 512)))
```




    VGGNet(
      Chain(
        VGGBlock(
          Chain(
            Conv((3, 3), 1 => 64, relu, pad=1),  [90m# 640 parameters[39m
            Conv((3, 3), 64 => 64, relu, pad=1),  [90m# 36_928 parameters[39m
            MaxPool((2, 2)),
          ),
        ),
        VGGBlock(
          Chain(
            Conv((3, 3), 64 => 128, relu, pad=1),  [90m# 73_856 parameters[39m
            Conv((3, 3), 128 => 128, relu, pad=1),  [90m# 147_584 parameters[39m
            MaxPool((2, 2)),
          ),
        ),
        VGGBlock(
          Chain(
            Conv((3, 3), 128 => 256, relu, pad=1),  [90m# 295_168 parameters[39m
            Conv((3, 3), 256 => 256, relu, pad=1),  [90m# 590_080 parameters[39m
            Conv((3, 3), 256 => 256, relu, pad=1),  [90m# 590_080 parameters[39m
            MaxPool((2, 2)),
          ),
        ),
        VGGBlock(
          Chain(
            Conv((3, 3), 256 => 512, relu, pad=1),  [90m# 1_180_160 parameters[39m
            Conv((3, 3), 512 => 512, relu, pad=1),  [90m# 2_359_808 parameters[39m
            Conv((3, 3), 512 => 512, relu, pad=1),  [90m# 2_359_808 parameters[39m
            MaxPool((2, 2)),
          ),
        ),
        VGGBlock(
          Chain(
            Conv((3, 3), 512 => 512, relu, pad=1),  [90m# 2_359_808 parameters[39m
            Conv((3, 3), 512 => 512, relu, pad=1),  [90m# 2_359_808 parameters[39m
            Conv((3, 3), 512 => 512, relu, pad=1),  [90m# 2_359_808 parameters[39m
            MaxPool((2, 2)),
          ),
        ),
        Flux.flatten,
        Dense(25088 => 4096, relu),         [90m# 102_764_544 parameters[39m
        Dropout(0.5),
        Dense(4096 => 4096, relu),          [90m# 16_781_312 parameters[39m
        Dropout(0.5),
        Dense(4096 => 10),                  [90m# 40_970 parameters[39m
        NNlib.softmax,
      ),
    ) [90m                  # Total: 32 arrays, [39m134_300_362 parameters, 512.318 MiB.




```julia
vgg_19 = VGGNet(((2, 64), (2, 128), (4, 256), (4, 512), (4, 512)))
```




    VGGNet(
      Chain(
        VGGBlock(
          Chain(
            Conv((3, 3), 1 => 64, relu, pad=1),  [90m# 640 parameters[39m
            Conv((3, 3), 64 => 64, relu, pad=1),  [90m# 36_928 parameters[39m
            MaxPool((2, 2)),
          ),
        ),
        VGGBlock(
          Chain(
            Conv((3, 3), 64 => 128, relu, pad=1),  [90m# 73_856 parameters[39m
            Conv((3, 3), 128 => 128, relu, pad=1),  [90m# 147_584 parameters[39m
            MaxPool((2, 2)),
          ),
        ),
        VGGBlock(
          Chain(
            Conv((3, 3), 128 => 256, relu, pad=1),  [90m# 295_168 parameters[39m
            Conv((3, 3), 256 => 256, relu, pad=1),  [90m# 590_080 parameters[39m
            Conv((3, 3), 256 => 256, relu, pad=1),  [90m# 590_080 parameters[39m
            Conv((3, 3), 256 => 256, relu, pad=1),  [90m# 590_080 parameters[39m
            MaxPool((2, 2)),
          ),
        ),
        VGGBlock(
          Chain(
            Conv((3, 3), 256 => 512, relu, pad=1),  [90m# 1_180_160 parameters[39m
            Conv((3, 3), 512 => 512, relu, pad=1),  [90m# 2_359_808 parameters[39m
            Conv((3, 3), 512 => 512, relu, pad=1),  [90m# 2_359_808 parameters[39m
            Conv((3, 3), 512 => 512, relu, pad=1),  [90m# 2_359_808 parameters[39m
            MaxPool((2, 2)),
          ),
        ),
        VGGBlock(
          Chain(
            Conv((3, 3), 512 => 512, relu, pad=1),  [90m# 2_359_808 parameters[39m
            Conv((3, 3), 512 => 512, relu, pad=1),  [90m# 2_359_808 parameters[39m
            Conv((3, 3), 512 => 512, relu, pad=1),  [90m# 2_359_808 parameters[39m
            Conv((3, 3), 512 => 512, relu, pad=1),  [90m# 2_359_808 parameters[39m
            MaxPool((2, 2)),
          ),
        ),
        Flux.flatten,
        Dense(25088 => 4096, relu),         [90m# 102_764_544 parameters[39m
        Dropout(0.5),
        Dense(4096 => 4096, relu),          [90m# 16_781_312 parameters[39m
        Dropout(0.5),
        Dense(4096 => 10),                  [90m# 40_970 parameters[39m
        NNlib.softmax,
      ),
    ) [90m                  # Total: 38 arrays, [39m139_610_058 parameters, 532.574 MiB.



**4**:


```julia
struct VGGSmallNet{N} <: AbstractClassifier 
    net::N
end 

function VGGSmallNet(; num_classes = 10)
    net = Flux.@autosize (56, 56, 1, 1) Chain(
        Conv((3,3), 1 => 16, relu, pad = 1),
        MaxPool((2,2), stride = 2),

        Conv((3,3), 16 => 32, relu, pad = 1),
        MaxPool((2,2), stride = 2),

        Conv((3,3), 32 => 64, relu, pad = 1),
        Conv((1,1), 64 => 64, relu),
        MaxPool((2,2), stride = 1),

        Conv((3,3), 64 => 128, relu, pad = 1),
        Conv((1,1), 128 => 128, relu),
        MaxPool((2,2), stride = 2),

        
        Flux.flatten,
        Dense(_ => 4096, relu),
        Dropout(0.5),
        Dense(4096, 4096, relu),
        Dropout(0.5),
        Dense(4096, num_classes),
        softmax
        
        
    )
    VGGSmallNet(net)
end
Flux.@layer VGGSmallNet 
(v::VGGSmallNet)(x) = v.net(x)

```


```julia
data_downsampled = d2lai.FashionMNISTData(batchsize = 128, resize = (56, 56))
model_small = VGGSmallNet()
opt = Descent(0.01)
trainer_small = Trainer(model_small, data_downsampled, opt; max_epochs = 10, gpu = true, board_yscale = :identity)
d2lai.fit(trainer_small);
```

    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mTrain Loss: 0.95945525, Val Loss: 0.5622649, Val Acc: 0.8125
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mTrain Loss: 0.56179816, Val Loss: 0.40773425, Val Acc: 0.875
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mTrain Loss: 0.5326469, Val Loss: 0.33665746, Val Acc: 0.8125
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mTrain Loss: 0.47002974, Val Loss: 0.24386045, Val Acc: 0.875
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mTrain Loss: 0.34498596, Val Loss: 0.2215715, Val Acc: 0.9375
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mTrain Loss: 0.51154643, Val Loss: 0.28165364, Val Acc: 0.875
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mTrain Loss: 0.25447252, Val Loss: 0.24042852, Val Acc: 0.875
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mTrain Loss: 0.40097007, Val Loss: 0.17593385, Val Acc: 0.9375
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mTrain Loss: 0.22108968, Val Loss: 0.22856621, Val Acc: 0.875
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mTrain Loss: 0.20709628, Val Loss: 0.19818446, Val Acc: 0.875



<?xml version="1.0" encoding="utf-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="600" height="400" viewBox="0 0 2400 1600">
<defs>
  <clipPath id="clip200">
    <rect x="0" y="0" width="2400" height="1600"/>
  </clipPath>
</defs>
<path clip-path="url(#clip200)" d="M0 1600 L2400 1600 L2400 0 L0 0  Z" fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<defs>
  <clipPath id="clip201">
    <rect x="480" y="0" width="1681" height="1600"/>
  </clipPath>
</defs>
<path clip-path="url(#clip200)" d="M186.274 1423.18 L2352.76 1423.18 L2352.76 47.2441 L186.274 47.2441  Z" fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<defs>
  <clipPath id="clip202">
    <rect x="186" y="47" width="2167" height="1377"/>
  </clipPath>
</defs>
<polyline clip-path="url(#clip202)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="474.684,1423.18 474.684,47.2441 "/>
<polyline clip-path="url(#clip202)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="928.873,1423.18 928.873,47.2441 "/>
<polyline clip-path="url(#clip202)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="1383.06,1423.18 1383.06,47.2441 "/>
<polyline clip-path="url(#clip202)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="1837.25,1423.18 1837.25,47.2441 "/>
<polyline clip-path="url(#clip202)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="2291.44,1423.18 2291.44,47.2441 "/>
<polyline clip-path="url(#clip202)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="186.274,1200.66 2352.76,1200.66 "/>
<polyline clip-path="url(#clip202)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="186.274,962.848 2352.76,962.848 "/>
<polyline clip-path="url(#clip202)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="186.274,725.04 2352.76,725.04 "/>
<polyline clip-path="url(#clip202)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="186.274,487.233 2352.76,487.233 "/>
<polyline clip-path="url(#clip202)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="186.274,249.426 2352.76,249.426 "/>
<polyline clip-path="url(#clip200)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="186.274,1423.18 2352.76,1423.18 "/>
<polyline clip-path="url(#clip200)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="474.684,1423.18 474.684,1404.28 "/>
<polyline clip-path="url(#clip200)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="928.873,1423.18 928.873,1404.28 "/>
<polyline clip-path="url(#clip200)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1383.06,1423.18 1383.06,1404.28 "/>
<polyline clip-path="url(#clip200)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1837.25,1423.18 1837.25,1404.28 "/>
<polyline clip-path="url(#clip200)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="2291.44,1423.18 2291.44,1404.28 "/>
<path clip-path="url(#clip200)" d="M469.337 1481.64 L485.656 1481.64 L485.656 1485.58 L463.712 1485.58 L463.712 1481.64 Q466.374 1478.89 470.957 1474.26 Q475.564 1469.61 476.744 1468.27 Q478.99 1465.74 479.869 1464.01 Q480.772 1462.25 480.772 1460.56 Q480.772 1457.8 478.828 1456.07 Q476.906 1454.33 473.804 1454.33 Q471.605 1454.33 469.152 1455.09 Q466.721 1455.86 463.943 1457.41 L463.943 1452.69 Q466.767 1451.55 469.221 1450.97 Q471.675 1450.39 473.712 1450.39 Q479.082 1450.39 482.277 1453.08 Q485.471 1455.77 485.471 1460.26 Q485.471 1462.39 484.661 1464.31 Q483.874 1466.2 481.767 1468.8 Q481.189 1469.47 478.087 1472.69 Q474.985 1475.88 469.337 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip200)" d="M931.882 1455.09 L920.077 1473.54 L931.882 1473.54 L931.882 1455.09 M930.656 1451.02 L936.535 1451.02 L936.535 1473.54 L941.466 1473.54 L941.466 1477.43 L936.535 1477.43 L936.535 1485.58 L931.882 1485.58 L931.882 1477.43 L916.281 1477.43 L916.281 1472.92 L930.656 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip200)" d="M1383.47 1466.44 Q1380.32 1466.44 1378.47 1468.59 Q1376.64 1470.74 1376.64 1474.49 Q1376.64 1478.22 1378.47 1480.39 Q1380.32 1482.55 1383.47 1482.55 Q1386.62 1482.55 1388.44 1480.39 Q1390.3 1478.22 1390.3 1474.49 Q1390.3 1470.74 1388.44 1468.59 Q1386.62 1466.44 1383.47 1466.44 M1392.75 1451.78 L1392.75 1456.04 Q1390.99 1455.21 1389.18 1454.77 Q1387.4 1454.33 1385.64 1454.33 Q1381.01 1454.33 1378.56 1457.45 Q1376.13 1460.58 1375.78 1466.9 Q1377.15 1464.89 1379.21 1463.82 Q1381.27 1462.73 1383.75 1462.73 Q1388.95 1462.73 1391.96 1465.9 Q1395 1469.05 1395 1474.49 Q1395 1479.82 1391.85 1483.03 Q1388.7 1486.25 1383.47 1486.25 Q1377.47 1486.25 1374.3 1481.67 Q1371.13 1477.06 1371.13 1468.33 Q1371.13 1460.14 1375.02 1455.28 Q1378.91 1450.39 1385.46 1450.39 Q1387.22 1450.39 1389 1450.74 Q1390.81 1451.09 1392.75 1451.78 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip200)" d="M1837.25 1469.17 Q1833.92 1469.17 1832 1470.95 Q1830.1 1472.73 1830.1 1475.86 Q1830.1 1478.98 1832 1480.77 Q1833.92 1482.55 1837.25 1482.55 Q1840.58 1482.55 1842.51 1480.77 Q1844.43 1478.96 1844.43 1475.86 Q1844.43 1472.73 1842.51 1470.95 Q1840.61 1469.17 1837.25 1469.17 M1832.58 1467.18 Q1829.57 1466.44 1827.88 1464.38 Q1826.21 1462.32 1826.21 1459.35 Q1826.21 1455.21 1829.15 1452.8 Q1832.11 1450.39 1837.25 1450.39 Q1842.41 1450.39 1845.35 1452.8 Q1848.29 1455.21 1848.29 1459.35 Q1848.29 1462.32 1846.6 1464.38 Q1844.94 1466.44 1841.95 1467.18 Q1845.33 1467.96 1847.2 1470.26 Q1849.1 1472.55 1849.1 1475.86 Q1849.1 1480.88 1846.02 1483.57 Q1842.97 1486.25 1837.25 1486.25 Q1831.53 1486.25 1828.46 1483.57 Q1825.4 1480.88 1825.4 1475.86 Q1825.4 1472.55 1827.3 1470.26 Q1829.2 1467.96 1832.58 1467.18 M1830.86 1459.79 Q1830.86 1462.48 1832.53 1463.98 Q1834.22 1465.49 1837.25 1465.49 Q1840.26 1465.49 1841.95 1463.98 Q1843.66 1462.48 1843.66 1459.79 Q1843.66 1457.11 1841.95 1455.6 Q1840.26 1454.1 1837.25 1454.1 Q1834.22 1454.1 1832.53 1455.6 Q1830.86 1457.11 1830.86 1459.79 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip200)" d="M2266.13 1481.64 L2273.77 1481.64 L2273.77 1455.28 L2265.46 1456.95 L2265.46 1452.69 L2273.72 1451.02 L2278.4 1451.02 L2278.4 1481.64 L2286.04 1481.64 L2286.04 1485.58 L2266.13 1485.58 L2266.13 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip200)" d="M2305.48 1454.1 Q2301.87 1454.1 2300.04 1457.66 Q2298.23 1461.2 2298.23 1468.33 Q2298.23 1475.44 2300.04 1479.01 Q2301.87 1482.55 2305.48 1482.55 Q2309.11 1482.55 2310.92 1479.01 Q2312.75 1475.44 2312.75 1468.33 Q2312.75 1461.2 2310.92 1457.66 Q2309.11 1454.1 2305.48 1454.1 M2305.48 1450.39 Q2311.29 1450.39 2314.35 1455 Q2317.42 1459.58 2317.42 1468.33 Q2317.42 1477.06 2314.35 1481.67 Q2311.29 1486.25 2305.48 1486.25 Q2299.67 1486.25 2296.59 1481.67 Q2293.54 1477.06 2293.54 1468.33 Q2293.54 1459.58 2296.59 1455 Q2299.67 1450.39 2305.48 1450.39 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip200)" d="M1189.7 1548.76 L1189.7 1551.62 L1162.78 1551.62 Q1163.16 1557.67 1166.41 1560.85 Q1169.68 1564 1175.51 1564 Q1178.88 1564 1182.03 1563.17 Q1185.22 1562.35 1188.34 1560.69 L1188.34 1566.23 Q1185.19 1567.57 1181.88 1568.27 Q1178.56 1568.97 1175.16 1568.97 Q1166.63 1568.97 1161.63 1564 Q1156.67 1559.04 1156.67 1550.57 Q1156.67 1541.82 1161.38 1536.69 Q1166.12 1531.54 1174.14 1531.54 Q1181.33 1531.54 1185.5 1536.18 Q1189.7 1540.8 1189.7 1548.76 M1183.85 1547.04 Q1183.78 1542.23 1181.14 1539.37 Q1178.53 1536.5 1174.2 1536.5 Q1169.3 1536.5 1166.34 1539.27 Q1163.41 1542.04 1162.97 1547.07 L1183.85 1547.04 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip200)" d="M1204.98 1562.7 L1204.98 1581.6 L1199.09 1581.6 L1199.09 1532.4 L1204.98 1532.4 L1204.98 1537.81 Q1206.83 1534.62 1209.63 1533.1 Q1212.46 1531.54 1216.38 1531.54 Q1222.87 1531.54 1226.91 1536.69 Q1230.99 1541.85 1230.99 1550.25 Q1230.99 1558.65 1226.91 1563.81 Q1222.87 1568.97 1216.38 1568.97 Q1212.46 1568.97 1209.63 1567.44 Q1206.83 1565.88 1204.98 1562.7 M1224.91 1550.25 Q1224.91 1543.79 1222.23 1540.13 Q1219.59 1536.44 1214.94 1536.44 Q1210.3 1536.44 1207.62 1540.13 Q1204.98 1543.79 1204.98 1550.25 Q1204.98 1556.71 1207.62 1560.4 Q1210.3 1564.07 1214.94 1564.07 Q1219.59 1564.07 1222.23 1560.4 Q1224.91 1556.71 1224.91 1550.25 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip200)" d="M1254.51 1536.5 Q1249.8 1536.5 1247.06 1540.19 Q1244.32 1543.85 1244.32 1550.25 Q1244.32 1556.65 1247.03 1560.34 Q1249.77 1564 1254.51 1564 Q1259.19 1564 1261.92 1560.31 Q1264.66 1556.62 1264.66 1550.25 Q1264.66 1543.92 1261.92 1540.23 Q1259.19 1536.5 1254.51 1536.5 M1254.51 1531.54 Q1262.15 1531.54 1266.51 1536.5 Q1270.87 1541.47 1270.87 1550.25 Q1270.87 1559 1266.51 1564 Q1262.15 1568.97 1254.51 1568.97 Q1246.84 1568.97 1242.48 1564 Q1238.15 1559 1238.15 1550.25 Q1238.15 1541.47 1242.48 1536.5 Q1246.84 1531.54 1254.51 1531.54 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip200)" d="M1306.23 1533.76 L1306.23 1539.24 Q1303.75 1537.87 1301.23 1537.2 Q1298.75 1536.5 1296.2 1536.5 Q1290.51 1536.5 1287.35 1540.13 Q1284.2 1543.73 1284.2 1550.25 Q1284.2 1556.78 1287.35 1560.4 Q1290.51 1564 1296.2 1564 Q1298.75 1564 1301.23 1563.33 Q1303.75 1562.63 1306.23 1561.26 L1306.23 1566.68 Q1303.78 1567.82 1301.14 1568.39 Q1298.53 1568.97 1295.57 1568.97 Q1287.51 1568.97 1282.77 1563.91 Q1278.03 1558.85 1278.03 1550.25 Q1278.03 1541.53 1282.8 1536.53 Q1287.61 1531.54 1295.95 1531.54 Q1298.65 1531.54 1301.23 1532.11 Q1303.81 1532.65 1306.23 1533.76 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip200)" d="M1346.05 1546.53 L1346.05 1568.04 L1340.19 1568.04 L1340.19 1546.72 Q1340.19 1541.66 1338.22 1539.14 Q1336.24 1536.63 1332.3 1536.63 Q1327.55 1536.63 1324.82 1539.65 Q1322.08 1542.68 1322.08 1547.9 L1322.08 1568.04 L1316.19 1568.04 L1316.19 1518.52 L1322.08 1518.52 L1322.08 1537.93 Q1324.18 1534.72 1327.01 1533.13 Q1329.88 1531.54 1333.6 1531.54 Q1339.74 1531.54 1342.9 1535.36 Q1346.05 1539.14 1346.05 1546.53 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip200)" d="M1380.45 1533.45 L1380.45 1538.98 Q1377.97 1537.71 1375.3 1537.07 Q1372.62 1536.44 1369.76 1536.44 Q1365.4 1536.44 1363.2 1537.77 Q1361.04 1539.11 1361.04 1541.79 Q1361.04 1543.82 1362.6 1545 Q1364.16 1546.15 1368.87 1547.2 L1370.87 1547.64 Q1377.11 1548.98 1379.72 1551.43 Q1382.36 1553.85 1382.36 1558.21 Q1382.36 1563.17 1378.42 1566.07 Q1374.5 1568.97 1367.63 1568.97 Q1364.76 1568.97 1361.64 1568.39 Q1358.56 1567.85 1355.12 1566.74 L1355.12 1560.69 Q1358.36 1562.38 1361.52 1563.24 Q1364.67 1564.07 1367.75 1564.07 Q1371.89 1564.07 1374.12 1562.66 Q1376.35 1561.23 1376.35 1558.65 Q1376.35 1556.27 1374.72 1554.99 Q1373.13 1553.72 1367.69 1552.54 L1365.65 1552.07 Q1360.21 1550.92 1357.79 1548.56 Q1355.37 1546.18 1355.37 1542.04 Q1355.37 1537.01 1358.94 1534.27 Q1362.5 1531.54 1369.06 1531.54 Q1372.31 1531.54 1375.17 1532.01 Q1378.03 1532.49 1380.45 1533.45 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><polyline clip-path="url(#clip200)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="186.274,1423.18 186.274,47.2441 "/>
<polyline clip-path="url(#clip200)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="186.274,1200.66 205.172,1200.66 "/>
<polyline clip-path="url(#clip200)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="186.274,962.848 205.172,962.848 "/>
<polyline clip-path="url(#clip200)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="186.274,725.04 205.172,725.04 "/>
<polyline clip-path="url(#clip200)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="186.274,487.233 205.172,487.233 "/>
<polyline clip-path="url(#clip200)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="186.274,249.426 205.172,249.426 "/>
<path clip-path="url(#clip200)" d="M62.9365 1186.45 Q59.3254 1186.45 57.4967 1190.02 Q55.6912 1193.56 55.6912 1200.69 Q55.6912 1207.8 57.4967 1211.36 Q59.3254 1214.9 62.9365 1214.9 Q66.5707 1214.9 68.3763 1211.36 Q70.205 1207.8 70.205 1200.69 Q70.205 1193.56 68.3763 1190.02 Q66.5707 1186.45 62.9365 1186.45 M62.9365 1182.75 Q68.7467 1182.75 71.8022 1187.36 Q74.8809 1191.94 74.8809 1200.69 Q74.8809 1209.42 71.8022 1214.02 Q68.7467 1218.61 62.9365 1218.61 Q57.1264 1218.61 54.0477 1214.02 Q50.9921 1209.42 50.9921 1200.69 Q50.9921 1191.94 54.0477 1187.36 Q57.1264 1182.75 62.9365 1182.75 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip200)" d="M83.0984 1212.06 L87.9827 1212.06 L87.9827 1217.94 L83.0984 1217.94 L83.0984 1212.06 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip200)" d="M98.2141 1183.38 L116.57 1183.38 L116.57 1187.31 L102.496 1187.31 L102.496 1195.78 Q103.515 1195.44 104.534 1195.27 Q105.552 1195.09 106.571 1195.09 Q112.358 1195.09 115.737 1198.26 Q119.117 1201.43 119.117 1206.85 Q119.117 1212.43 115.645 1215.53 Q112.172 1218.61 105.853 1218.61 Q103.677 1218.61 101.409 1218.24 Q99.1632 1217.87 96.7558 1217.13 L96.7558 1212.43 Q98.8391 1213.56 101.061 1214.12 Q103.284 1214.67 105.76 1214.67 Q109.765 1214.67 112.103 1212.57 Q114.441 1210.46 114.441 1206.85 Q114.441 1203.24 112.103 1201.13 Q109.765 1199.02 105.76 1199.02 Q103.885 1199.02 102.01 1199.44 Q100.159 1199.86 98.2141 1200.74 L98.2141 1183.38 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip200)" d="M138.33 1186.45 Q134.719 1186.45 132.89 1190.02 Q131.084 1193.56 131.084 1200.69 Q131.084 1207.8 132.89 1211.36 Q134.719 1214.9 138.33 1214.9 Q141.964 1214.9 143.769 1211.36 Q145.598 1207.8 145.598 1200.69 Q145.598 1193.56 143.769 1190.02 Q141.964 1186.45 138.33 1186.45 M138.33 1182.75 Q144.14 1182.75 147.195 1187.36 Q150.274 1191.94 150.274 1200.69 Q150.274 1209.42 147.195 1214.02 Q144.14 1218.61 138.33 1218.61 Q132.519 1218.61 129.441 1214.02 Q126.385 1209.42 126.385 1200.69 Q126.385 1191.94 129.441 1187.36 Q132.519 1182.75 138.33 1182.75 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip200)" d="M63.9319 948.647 Q60.3208 948.647 58.4921 952.211 Q56.6865 955.753 56.6865 962.883 Q56.6865 969.989 58.4921 973.554 Q60.3208 977.096 63.9319 977.096 Q67.5661 977.096 69.3717 973.554 Q71.2004 969.989 71.2004 962.883 Q71.2004 955.753 69.3717 952.211 Q67.5661 948.647 63.9319 948.647 M63.9319 944.943 Q69.742 944.943 72.7976 949.549 Q75.8763 954.133 75.8763 962.883 Q75.8763 971.609 72.7976 976.216 Q69.742 980.799 63.9319 980.799 Q58.1217 980.799 55.043 976.216 Q51.9875 971.609 51.9875 962.883 Q51.9875 954.133 55.043 949.549 Q58.1217 944.943 63.9319 944.943 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip200)" d="M84.0938 974.248 L88.978 974.248 L88.978 980.128 L84.0938 980.128 L84.0938 974.248 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip200)" d="M97.9826 945.568 L120.205 945.568 L120.205 947.559 L107.658 980.128 L102.774 980.128 L114.58 949.503 L97.9826 949.503 L97.9826 945.568 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip200)" d="M129.371 945.568 L147.728 945.568 L147.728 949.503 L133.654 949.503 L133.654 957.975 Q134.672 957.628 135.691 957.466 Q136.709 957.281 137.728 957.281 Q143.515 957.281 146.894 960.452 Q150.274 963.623 150.274 969.04 Q150.274 974.619 146.802 977.721 Q143.33 980.799 137.01 980.799 Q134.834 980.799 132.566 980.429 Q130.32 980.058 127.913 979.318 L127.913 974.619 Q129.996 975.753 132.219 976.309 Q134.441 976.864 136.918 976.864 Q140.922 976.864 143.26 974.758 Q145.598 972.651 145.598 969.04 Q145.598 965.429 143.26 963.322 Q140.922 961.216 136.918 961.216 Q135.043 961.216 133.168 961.633 Q131.316 962.049 129.371 962.929 L129.371 945.568 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip200)" d="M53.7467 738.385 L61.3856 738.385 L61.3856 712.02 L53.0754 713.686 L53.0754 709.427 L61.3393 707.76 L66.0152 707.76 L66.0152 738.385 L73.654 738.385 L73.654 742.32 L53.7467 742.32 L53.7467 738.385 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip200)" d="M83.0984 736.441 L87.9827 736.441 L87.9827 742.32 L83.0984 742.32 L83.0984 736.441 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip200)" d="M108.168 710.839 Q104.557 710.839 102.728 714.404 Q100.922 717.946 100.922 725.075 Q100.922 732.182 102.728 735.746 Q104.557 739.288 108.168 739.288 Q111.802 739.288 113.608 735.746 Q115.436 732.182 115.436 725.075 Q115.436 717.946 113.608 714.404 Q111.802 710.839 108.168 710.839 M108.168 707.135 Q113.978 707.135 117.033 711.742 Q120.112 716.325 120.112 725.075 Q120.112 733.802 117.033 738.408 Q113.978 742.992 108.168 742.992 Q102.358 742.992 99.2789 738.408 Q96.2234 733.802 96.2234 725.075 Q96.2234 716.325 99.2789 711.742 Q102.358 707.135 108.168 707.135 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip200)" d="M138.33 710.839 Q134.719 710.839 132.89 714.404 Q131.084 717.946 131.084 725.075 Q131.084 732.182 132.89 735.746 Q134.719 739.288 138.33 739.288 Q141.964 739.288 143.769 735.746 Q145.598 732.182 145.598 725.075 Q145.598 717.946 143.769 714.404 Q141.964 710.839 138.33 710.839 M138.33 707.135 Q144.14 707.135 147.195 711.742 Q150.274 716.325 150.274 725.075 Q150.274 733.802 147.195 738.408 Q144.14 742.992 138.33 742.992 Q132.519 742.992 129.441 738.408 Q126.385 733.802 126.385 725.075 Q126.385 716.325 129.441 711.742 Q132.519 707.135 138.33 707.135 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip200)" d="M54.7421 500.578 L62.381 500.578 L62.381 474.212 L54.0708 475.879 L54.0708 471.62 L62.3347 469.953 L67.0106 469.953 L67.0106 500.578 L74.6494 500.578 L74.6494 504.513 L54.7421 504.513 L54.7421 500.578 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip200)" d="M84.0938 498.633 L88.978 498.633 L88.978 504.513 L84.0938 504.513 L84.0938 498.633 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip200)" d="M103.191 500.578 L119.51 500.578 L119.51 504.513 L97.566 504.513 L97.566 500.578 Q100.228 497.823 104.811 493.194 Q109.418 488.541 110.598 487.198 Q112.844 484.675 113.723 482.939 Q114.626 481.18 114.626 479.49 Q114.626 476.735 112.682 474.999 Q110.76 473.263 107.658 473.263 Q105.459 473.263 103.006 474.027 Q100.575 474.791 97.7974 476.342 L97.7974 471.62 Q100.621 470.485 103.075 469.907 Q105.529 469.328 107.566 469.328 Q112.936 469.328 116.131 472.013 Q119.325 474.698 119.325 479.189 Q119.325 481.319 118.515 483.24 Q117.728 485.138 115.621 487.731 Q115.043 488.402 111.941 491.62 Q108.839 494.814 103.191 500.578 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip200)" d="M129.371 469.953 L147.728 469.953 L147.728 473.888 L133.654 473.888 L133.654 482.36 Q134.672 482.013 135.691 481.851 Q136.709 481.666 137.728 481.666 Q143.515 481.666 146.894 484.837 Q150.274 488.008 150.274 493.425 Q150.274 499.004 146.802 502.106 Q143.33 505.184 137.01 505.184 Q134.834 505.184 132.566 504.814 Q130.32 504.444 127.913 503.703 L127.913 499.004 Q129.996 500.138 132.219 500.694 Q134.441 501.249 136.918 501.249 Q140.922 501.249 143.26 499.143 Q145.598 497.036 145.598 493.425 Q145.598 489.814 143.26 487.708 Q140.922 485.601 136.918 485.601 Q135.043 485.601 133.168 486.018 Q131.316 486.434 129.371 487.314 L129.371 469.953 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip200)" d="M53.7467 262.77 L61.3856 262.77 L61.3856 236.405 L53.0754 238.071 L53.0754 233.812 L61.3393 232.146 L66.0152 232.146 L66.0152 262.77 L73.654 262.77 L73.654 266.706 L53.7467 266.706 L53.7467 262.77 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip200)" d="M83.0984 260.826 L87.9827 260.826 L87.9827 266.706 L83.0984 266.706 L83.0984 260.826 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip200)" d="M98.2141 232.146 L116.57 232.146 L116.57 236.081 L102.496 236.081 L102.496 244.553 Q103.515 244.206 104.534 244.044 Q105.552 243.858 106.571 243.858 Q112.358 243.858 115.737 247.03 Q119.117 250.201 119.117 255.618 Q119.117 261.196 115.645 264.298 Q112.172 267.377 105.853 267.377 Q103.677 267.377 101.409 267.006 Q99.1632 266.636 96.7558 265.895 L96.7558 261.196 Q98.8391 262.331 101.061 262.886 Q103.284 263.442 105.76 263.442 Q109.765 263.442 112.103 261.335 Q114.441 259.229 114.441 255.618 Q114.441 252.007 112.103 249.9 Q109.765 247.794 105.76 247.794 Q103.885 247.794 102.01 248.21 Q100.159 248.627 98.2141 249.507 L98.2141 232.146 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip200)" d="M138.33 235.224 Q134.719 235.224 132.89 238.789 Q131.084 242.331 131.084 249.46 Q131.084 256.567 132.89 260.132 Q134.719 263.673 138.33 263.673 Q141.964 263.673 143.769 260.132 Q145.598 256.567 145.598 249.46 Q145.598 242.331 143.769 238.789 Q141.964 235.224 138.33 235.224 M138.33 231.521 Q144.14 231.521 147.195 236.127 Q150.274 240.71 150.274 249.46 Q150.274 258.187 147.195 262.794 Q144.14 267.377 138.33 267.377 Q132.519 267.377 129.441 262.794 Q126.385 258.187 126.385 249.46 Q126.385 240.71 129.441 236.127 Q132.519 231.521 138.33 231.521 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><polyline clip-path="url(#clip202)" style="stroke:#009af9; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="247.59,86.1857 474.684,1032.08 701.779,1160.44 928.873,1235.55 1155.97,1280.92 1383.06,1312.79 1610.16,1336.97 1837.25,1355.92 2064.35,1370.36 2291.44,1384.24 "/>
<polyline clip-path="url(#clip202)" style="stroke:#e26f46; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="247.59,872.55 474.684,1027.21 701.779,1111.92 928.873,1194.64 1155.97,1201.01 1383.06,1239.53 1610.16,1279.12 1837.25,1273.27 2064.35,1305.68 2291.44,1307.21 "/>
<polyline clip-path="url(#clip202)" style="stroke:#3da44d; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="247.59,1024.09 474.684,965.2 701.779,947.891 928.873,898.881 1155.97,903.02 1383.06,882.983 1610.16,869.249 1837.25,867.556 2064.35,856.926 2291.44,859.936 "/>
<path clip-path="url(#clip200)" d="M1841.86 300.469 L2280.54 300.469 L2280.54 93.1086 L1841.86 93.1086  Z" fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<polyline clip-path="url(#clip200)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1841.86,300.469 2280.54,300.469 2280.54,93.1086 1841.86,93.1086 1841.86,300.469 "/>
<polyline clip-path="url(#clip200)" style="stroke:#009af9; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1865.93,144.949 2010.36,144.949 "/>
<path clip-path="url(#clip200)" d="M2041.84 128.942 L2041.84 136.303 L2050.61 136.303 L2050.61 139.613 L2041.84 139.613 L2041.84 153.687 Q2041.84 156.858 2042.7 157.761 Q2043.58 158.664 2046.24 158.664 L2050.61 158.664 L2050.61 162.229 L2046.24 162.229 Q2041.31 162.229 2039.43 160.4 Q2037.56 158.548 2037.56 153.687 L2037.56 139.613 L2034.43 139.613 L2034.43 136.303 L2037.56 136.303 L2037.56 128.942 L2041.84 128.942 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip200)" d="M2071.24 140.284 Q2070.52 139.868 2069.66 139.682 Q2068.83 139.474 2067.81 139.474 Q2064.2 139.474 2062.26 141.835 Q2060.33 144.173 2060.33 148.571 L2060.33 162.229 L2056.05 162.229 L2056.05 136.303 L2060.33 136.303 L2060.33 140.331 Q2061.68 137.969 2063.83 136.835 Q2065.98 135.678 2069.06 135.678 Q2069.5 135.678 2070.03 135.747 Q2070.57 135.794 2071.21 135.909 L2071.24 140.284 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip200)" d="M2087.49 149.196 Q2082.33 149.196 2080.33 150.377 Q2078.34 151.557 2078.34 154.405 Q2078.34 156.673 2079.83 158.016 Q2081.33 159.335 2083.9 159.335 Q2087.44 159.335 2089.57 156.835 Q2091.72 154.312 2091.72 150.145 L2091.72 149.196 L2087.49 149.196 M2095.98 147.437 L2095.98 162.229 L2091.72 162.229 L2091.72 158.293 Q2090.27 160.655 2088.09 161.789 Q2085.91 162.9 2082.77 162.9 Q2078.78 162.9 2076.42 160.678 Q2074.08 158.432 2074.08 154.682 Q2074.08 150.307 2077 148.085 Q2079.94 145.863 2085.75 145.863 L2091.72 145.863 L2091.72 145.446 Q2091.72 142.507 2089.78 140.909 Q2087.86 139.289 2084.36 139.289 Q2082.14 139.289 2080.03 139.821 Q2077.93 140.354 2075.98 141.419 L2075.98 137.483 Q2078.32 136.581 2080.52 136.141 Q2082.72 135.678 2084.8 135.678 Q2090.43 135.678 2093.2 138.594 Q2095.98 141.511 2095.98 147.437 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip200)" d="M2104.76 136.303 L2109.01 136.303 L2109.01 162.229 L2104.76 162.229 L2104.76 136.303 M2104.76 126.21 L2109.01 126.21 L2109.01 131.604 L2104.76 131.604 L2104.76 126.21 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip200)" d="M2139.48 146.581 L2139.48 162.229 L2135.22 162.229 L2135.22 146.719 Q2135.22 143.039 2133.78 141.21 Q2132.35 139.382 2129.48 139.382 Q2126.03 139.382 2124.04 141.581 Q2122.05 143.78 2122.05 147.576 L2122.05 162.229 L2117.76 162.229 L2117.76 136.303 L2122.05 136.303 L2122.05 140.331 Q2123.58 137.993 2125.64 136.835 Q2127.72 135.678 2130.43 135.678 Q2134.89 135.678 2137.19 138.456 Q2139.48 141.21 2139.48 146.581 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip200)" d="M2167.67 170.099 L2167.67 173.409 L2143.04 173.409 L2143.04 170.099 L2167.67 170.099 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip200)" d="M2171.68 126.21 L2175.94 126.21 L2175.94 162.229 L2171.68 162.229 L2171.68 126.21 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip200)" d="M2194.89 139.289 Q2191.47 139.289 2189.48 141.974 Q2187.49 144.636 2187.49 149.289 Q2187.49 153.942 2189.45 156.627 Q2191.45 159.289 2194.89 159.289 Q2198.3 159.289 2200.29 156.604 Q2202.28 153.918 2202.28 149.289 Q2202.28 144.682 2200.29 141.997 Q2198.3 139.289 2194.89 139.289 M2194.89 135.678 Q2200.45 135.678 2203.62 139.289 Q2206.79 142.9 2206.79 149.289 Q2206.79 155.655 2203.62 159.289 Q2200.45 162.9 2194.89 162.9 Q2189.32 162.9 2186.14 159.289 Q2183 155.655 2183 149.289 Q2183 142.9 2186.14 139.289 Q2189.32 135.678 2194.89 135.678 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip200)" d="M2230.38 137.067 L2230.38 141.094 Q2228.57 140.169 2226.63 139.706 Q2224.69 139.243 2222.6 139.243 Q2219.43 139.243 2217.83 140.215 Q2216.26 141.187 2216.26 143.131 Q2216.26 144.613 2217.39 145.469 Q2218.53 146.303 2221.95 147.067 L2223.41 147.391 Q2227.95 148.363 2229.85 150.145 Q2231.77 151.905 2231.77 155.076 Q2231.77 158.687 2228.9 160.793 Q2226.05 162.9 2221.05 162.9 Q2218.97 162.9 2216.7 162.483 Q2214.45 162.09 2211.95 161.28 L2211.95 156.881 Q2214.32 158.108 2216.61 158.733 Q2218.9 159.335 2221.14 159.335 Q2224.15 159.335 2225.77 158.317 Q2227.39 157.275 2227.39 155.4 Q2227.39 153.664 2226.21 152.738 Q2225.06 151.812 2221.1 150.956 L2219.62 150.608 Q2215.66 149.775 2213.9 148.062 Q2212.14 146.326 2212.14 143.317 Q2212.14 139.659 2214.73 137.669 Q2217.32 135.678 2222.09 135.678 Q2224.45 135.678 2226.54 136.025 Q2228.62 136.372 2230.38 137.067 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip200)" d="M2255.08 137.067 L2255.08 141.094 Q2253.27 140.169 2251.33 139.706 Q2249.38 139.243 2247.3 139.243 Q2244.13 139.243 2242.53 140.215 Q2240.96 141.187 2240.96 143.131 Q2240.96 144.613 2242.09 145.469 Q2243.23 146.303 2246.65 147.067 L2248.11 147.391 Q2252.65 148.363 2254.55 150.145 Q2256.47 151.905 2256.47 155.076 Q2256.47 158.687 2253.6 160.793 Q2250.75 162.9 2245.75 162.9 Q2243.67 162.9 2241.4 162.483 Q2239.15 162.09 2236.65 161.28 L2236.65 156.881 Q2239.01 158.108 2241.31 158.733 Q2243.6 159.335 2245.84 159.335 Q2248.85 159.335 2250.47 158.317 Q2252.09 157.275 2252.09 155.4 Q2252.09 153.664 2250.91 152.738 Q2249.75 151.812 2245.8 150.956 L2244.32 150.608 Q2240.36 149.775 2238.6 148.062 Q2236.84 146.326 2236.84 143.317 Q2236.84 139.659 2239.43 137.669 Q2242.02 135.678 2246.79 135.678 Q2249.15 135.678 2251.24 136.025 Q2253.32 136.372 2255.08 137.067 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><polyline clip-path="url(#clip200)" style="stroke:#e26f46; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1865.93,196.789 2010.36,196.789 "/>
<path clip-path="url(#clip200)" d="M2034.43 188.143 L2038.95 188.143 L2047.05 209.902 L2055.15 188.143 L2059.66 188.143 L2049.94 214.069 L2044.15 214.069 L2034.43 188.143 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip200)" d="M2077.33 201.036 Q2072.16 201.036 2070.17 202.217 Q2068.18 203.397 2068.18 206.245 Q2068.18 208.513 2069.66 209.856 Q2071.17 211.175 2073.74 211.175 Q2077.28 211.175 2079.41 208.675 Q2081.56 206.152 2081.56 201.985 L2081.56 201.036 L2077.33 201.036 M2085.82 199.277 L2085.82 214.069 L2081.56 214.069 L2081.56 210.133 Q2080.1 212.495 2077.93 213.629 Q2075.75 214.74 2072.6 214.74 Q2068.62 214.74 2066.26 212.518 Q2063.92 210.272 2063.92 206.522 Q2063.92 202.147 2066.84 199.925 Q2069.78 197.703 2075.59 197.703 L2081.56 197.703 L2081.56 197.286 Q2081.56 194.347 2079.62 192.749 Q2077.7 191.129 2074.2 191.129 Q2071.98 191.129 2069.87 191.661 Q2067.77 192.194 2065.82 193.259 L2065.82 189.323 Q2068.16 188.421 2070.36 187.981 Q2072.56 187.518 2074.64 187.518 Q2080.27 187.518 2083.04 190.434 Q2085.82 193.351 2085.82 199.277 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip200)" d="M2094.59 178.05 L2098.85 178.05 L2098.85 214.069 L2094.59 214.069 L2094.59 178.05 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip200)" d="M2127.46 221.939 L2127.46 225.249 L2102.83 225.249 L2102.83 221.939 L2127.46 221.939 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip200)" d="M2131.47 178.05 L2135.73 178.05 L2135.73 214.069 L2131.47 214.069 L2131.47 178.05 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip200)" d="M2154.69 191.129 Q2151.26 191.129 2149.27 193.814 Q2147.28 196.476 2147.28 201.129 Q2147.28 205.782 2149.25 208.467 Q2151.24 211.129 2154.69 211.129 Q2158.09 211.129 2160.08 208.444 Q2162.07 205.758 2162.07 201.129 Q2162.07 196.522 2160.08 193.837 Q2158.09 191.129 2154.69 191.129 M2154.69 187.518 Q2160.24 187.518 2163.41 191.129 Q2166.58 194.74 2166.58 201.129 Q2166.58 207.495 2163.41 211.129 Q2160.24 214.74 2154.69 214.74 Q2149.11 214.74 2145.94 211.129 Q2142.79 207.495 2142.79 201.129 Q2142.79 194.74 2145.94 191.129 Q2149.11 187.518 2154.69 187.518 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip200)" d="M2190.17 188.907 L2190.17 192.934 Q2188.37 192.009 2186.42 191.546 Q2184.48 191.083 2182.39 191.083 Q2179.22 191.083 2177.63 192.055 Q2176.05 193.027 2176.05 194.971 Q2176.05 196.453 2177.19 197.309 Q2178.32 198.143 2181.75 198.907 L2183.2 199.231 Q2187.74 200.203 2189.64 201.985 Q2191.56 203.745 2191.56 206.916 Q2191.56 210.527 2188.69 212.633 Q2185.84 214.74 2180.84 214.74 Q2178.76 214.74 2176.49 214.323 Q2174.25 213.93 2171.75 213.12 L2171.75 208.721 Q2174.11 209.948 2176.4 210.573 Q2178.69 211.175 2180.94 211.175 Q2183.95 211.175 2185.57 210.157 Q2187.19 209.115 2187.19 207.24 Q2187.19 205.504 2186.01 204.578 Q2184.85 203.652 2180.89 202.796 L2179.41 202.448 Q2175.45 201.615 2173.69 199.902 Q2171.93 198.166 2171.93 195.157 Q2171.93 191.499 2174.52 189.509 Q2177.12 187.518 2181.88 187.518 Q2184.25 187.518 2186.33 187.865 Q2188.41 188.212 2190.17 188.907 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip200)" d="M2214.87 188.907 L2214.87 192.934 Q2213.07 192.009 2211.12 191.546 Q2209.18 191.083 2207.09 191.083 Q2203.92 191.083 2202.32 192.055 Q2200.75 193.027 2200.75 194.971 Q2200.75 196.453 2201.88 197.309 Q2203.02 198.143 2206.44 198.907 L2207.9 199.231 Q2212.44 200.203 2214.34 201.985 Q2216.26 203.745 2216.26 206.916 Q2216.26 210.527 2213.39 212.633 Q2210.54 214.74 2205.54 214.74 Q2203.46 214.74 2201.19 214.323 Q2198.94 213.93 2196.44 213.12 L2196.44 208.721 Q2198.81 209.948 2201.1 210.573 Q2203.39 211.175 2205.63 211.175 Q2208.64 211.175 2210.26 210.157 Q2211.88 209.115 2211.88 207.24 Q2211.88 205.504 2210.7 204.578 Q2209.55 203.652 2205.59 202.796 L2204.11 202.448 Q2200.15 201.615 2198.39 199.902 Q2196.63 198.166 2196.63 195.157 Q2196.63 191.499 2199.22 189.509 Q2201.82 187.518 2206.58 187.518 Q2208.94 187.518 2211.03 187.865 Q2213.11 188.212 2214.87 188.907 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><polyline clip-path="url(#clip200)" style="stroke:#3da44d; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1865.93,248.629 2010.36,248.629 "/>
<path clip-path="url(#clip200)" d="M2034.43 239.983 L2038.95 239.983 L2047.05 261.742 L2055.15 239.983 L2059.66 239.983 L2049.94 265.909 L2044.15 265.909 L2034.43 239.983 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip200)" d="M2077.33 252.876 Q2072.16 252.876 2070.17 254.057 Q2068.18 255.237 2068.18 258.085 Q2068.18 260.353 2069.66 261.696 Q2071.17 263.015 2073.74 263.015 Q2077.28 263.015 2079.41 260.515 Q2081.56 257.992 2081.56 253.825 L2081.56 252.876 L2077.33 252.876 M2085.82 251.117 L2085.82 265.909 L2081.56 265.909 L2081.56 261.973 Q2080.1 264.335 2077.93 265.469 Q2075.75 266.58 2072.6 266.58 Q2068.62 266.58 2066.26 264.358 Q2063.92 262.112 2063.92 258.362 Q2063.92 253.987 2066.84 251.765 Q2069.78 249.543 2075.59 249.543 L2081.56 249.543 L2081.56 249.126 Q2081.56 246.187 2079.62 244.589 Q2077.7 242.969 2074.2 242.969 Q2071.98 242.969 2069.87 243.501 Q2067.77 244.034 2065.82 245.099 L2065.82 241.163 Q2068.16 240.261 2070.36 239.821 Q2072.56 239.358 2074.64 239.358 Q2080.27 239.358 2083.04 242.274 Q2085.82 245.191 2085.82 251.117 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip200)" d="M2094.59 229.89 L2098.85 229.89 L2098.85 265.909 L2094.59 265.909 L2094.59 229.89 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip200)" d="M2127.46 273.779 L2127.46 277.089 L2102.83 277.089 L2102.83 273.779 L2127.46 273.779 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip200)" d="M2143.25 252.876 Q2138.09 252.876 2136.1 254.057 Q2134.11 255.237 2134.11 258.085 Q2134.11 260.353 2135.59 261.696 Q2137.09 263.015 2139.66 263.015 Q2143.2 263.015 2145.33 260.515 Q2147.49 257.992 2147.49 253.825 L2147.49 252.876 L2143.25 252.876 M2151.75 251.117 L2151.75 265.909 L2147.49 265.909 L2147.49 261.973 Q2146.03 264.335 2143.85 265.469 Q2141.68 266.58 2138.53 266.58 Q2134.55 266.58 2132.19 264.358 Q2129.85 262.112 2129.85 258.362 Q2129.85 253.987 2132.76 251.765 Q2135.7 249.543 2141.51 249.543 L2147.49 249.543 L2147.49 249.126 Q2147.49 246.187 2145.54 244.589 Q2143.62 242.969 2140.13 242.969 Q2137.9 242.969 2135.8 243.501 Q2133.69 244.034 2131.75 245.099 L2131.75 241.163 Q2134.08 240.261 2136.28 239.821 Q2138.48 239.358 2140.57 239.358 Q2146.19 239.358 2148.97 242.274 Q2151.75 245.191 2151.75 251.117 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip200)" d="M2179.18 240.978 L2179.18 244.96 Q2177.37 243.964 2175.54 243.478 Q2173.74 242.969 2171.88 242.969 Q2167.74 242.969 2165.45 245.608 Q2163.16 248.224 2163.16 252.969 Q2163.16 257.714 2165.45 260.353 Q2167.74 262.969 2171.88 262.969 Q2173.74 262.969 2175.54 262.483 Q2177.37 261.973 2179.18 260.978 L2179.18 264.913 Q2177.39 265.747 2175.47 266.163 Q2173.57 266.58 2171.42 266.58 Q2165.57 266.58 2162.12 262.899 Q2158.67 259.219 2158.67 252.969 Q2158.67 246.626 2162.14 242.992 Q2165.63 239.358 2171.7 239.358 Q2173.67 239.358 2175.54 239.775 Q2177.42 240.168 2179.18 240.978 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip200)" d="M2205.24 240.978 L2205.24 244.96 Q2203.44 243.964 2201.61 243.478 Q2199.8 242.969 2197.95 242.969 Q2193.81 242.969 2191.51 245.608 Q2189.22 248.224 2189.22 252.969 Q2189.22 257.714 2191.51 260.353 Q2193.81 262.969 2197.95 262.969 Q2199.8 262.969 2201.61 262.483 Q2203.44 261.973 2205.24 260.978 L2205.24 264.913 Q2203.46 265.747 2201.54 266.163 Q2199.64 266.58 2197.49 266.58 Q2191.63 266.58 2188.18 262.899 Q2184.73 259.219 2184.73 252.969 Q2184.73 246.626 2188.2 242.992 Q2191.7 239.358 2197.76 239.358 Q2199.73 239.358 2201.61 239.775 Q2203.48 240.168 2205.24 240.978 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /></svg>




```julia

```
