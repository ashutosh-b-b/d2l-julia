using Documenter 

mathengine = MathJax3(Dict(:loader => Dict("load" => ["[tex]/require", "[tex]/mathtools"]),
    :tex => Dict("inlineMath" => [["\$", "\$"], ["\\(", "\\)"]],
        "packages" => [
            "base",
            "ams",
            "autoload",
            "mathtools",
            "require",
        ])))

makedocs(;
    authors = "Ashutosh Bharambe",
    sitename = "d2l Julia",
    remotes = nothing,
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", nothing) == "true",
        assets = String[],
        mathengine,
        edit_link = nothing),
    pagesonly = true,
    doctest = true,
    linkcheck = true,
    warnonly = [:missing_docs],
    pages = [
        "5. Convolutional Neural Networks" => [
            "CH5.Convolutional_Neural_Networks/CNN_2.md",
            "CH5.Convolutional_Neural_Networks/CNN_3.md",
            "CH5.Convolutional_Neural_Networks/CNN_4.md",
            "CH5.Convolutional_Neural_Networks/CNN_5.md",
            "CH5.Convolutional_Neural_Networks/CNN_6.md",
        ],
        "6. Modern Convolutional Neural Networks" => [
            "CH6.ModernConvolutionalNeuralNetworks/MCNN_0.md",
            "CH6.ModernConvolutionalNeuralNetworks/MCNN_1.md",
            "CH6.ModernConvolutionalNeuralNetworks/MCNN_2.md",
            "CH6.ModernConvolutionalNeuralNetworks/MCNN_3.md",
            "CH6.ModernConvolutionalNeuralNetworks/MCNN_4.md",
            "CH6.ModernConvolutionalNeuralNetworks/MCNN_5.md",
            "CH6.ModernConvolutionalNeuralNetworks/MCNN_6.md",
            "CH6.ModernConvolutionalNeuralNetworks/MCNN_7.md",
            "CH6.ModernConvolutionalNeuralNetworks/MCNN_8.md",
        ],
        "7. Recurrent Neural Networks" => [
            "CH7.Recurrent_Neural_Networks/RNN_0.md",
            "CH7.Recurrent_Neural_Networks/RNN_1.md",
            "CH7.Recurrent_Neural_Networks/RNN_2.md",
            "CH7.Recurrent_Neural_Networks/RNN_3.md",
            "CH7.Recurrent_Neural_Networks/RNN_4.md",
            "CH7.Recurrent_Neural_Networks/RNN_5.md",
            "CH7.Recurrent_Neural_Networks/RNN_6.md",
            "CH7.Recurrent_Neural_Networks/RNN_7.md",
        ],
        "8. Modern Recurrent Neural Networks" => [
            "CH8.Modern_Recurrent_Neural_Networks/MRNN_1.md",
            "CH8.Modern_Recurrent_Neural_Networks/MRNN_2.md",
            "CH8.Modern_Recurrent_Neural_Networks/MRNN_3.md",
            "CH8.Modern_Recurrent_Neural_Networks/MRNN_4.md",
            "CH8.Modern_Recurrent_Neural_Networks/MRNN_5.md",
            "CH8.Modern_Recurrent_Neural_Networks/MRNN_6.md",
            "CH8.Modern_Recurrent_Neural_Networks/MRNN7.md",
        ],
        "9. Attention Mechanisms and Transformers" => [
            "CH9.Attention_Mechanisms_and_Transformers/ATTN_1.md",
            "CH9.Attention_Mechanisms_and_Transformers/ATTN_2.md",
            "CH9.Attention_Mechanisms_and_Transformers/ATTN_3.md",
            "CH9.Attention_Mechanisms_and_Transformers/ATTN_4.md",
            "CH9.Attention_Mechanisms_and_Transformers/ATTN_5.md",
            "CH9.Attention_Mechanisms_and_Transformers/ATTN_6.md",
        ],
    ])