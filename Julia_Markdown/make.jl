using Documenter 
using DocumenterVitepress
using DocumenterCitations

mathengine = MathJax3(Dict(:loader => Dict("load" => ["[tex]/require", "[tex]/mathtools"]),
    :tex => Dict("inlineMath" => [["\$", "\$"], ["\\(", "\\)"]],
        "packages" => [
            "base",
            "ams",
            "autoload",
            "mathtools",
            "require",
        ])))


chapter_titles = [
    "Linear Neural Networks for Regression",
    "Linear Neural Networks for Classification",
    "Multilayer Perceptron",
    "Convolutional Neural Networks",
    "Modern Convolutional Neural Networks",
    "Recurrent Neural Networks",
    "Modern Recurrent Neural Networks",
    "Attention Mechanisms and Transformers"
]

chapter_folders = [
    
 "CH3.Linear_Regression",
 "CH4.Linear_Classification",
 "CH5.MLP",
 "CH6.Convolutional_Neural_Networks",
 "CH7.ModernConvolutionalNeuralNetworks",
 "CH8.Recurrent_Neural_Networks",
 "CH9.Modern_Recurrent_Neural_Networks",
 "CH10.Attention_Mechanisms_and_Transformers",
]

pages = map(chapter_titles, chapter_folders) do title, folder
    rel_folder = joinpath(@__DIR__, "src", folder)
    md_files = filter(f -> endswith(f, ".md"), readdir(rel_folder; join=false, sort=true))
    title => map(f -> joinpath(folder, f), md_files)
end

bib = CitationBibliography(
    joinpath(@__DIR__, "src", "ref.bib");
    style=:numeric
)

makedocs(
    sitename = "d2l Julia",
    authors = "Ashutosh Bharambe",
    # format = Documenter.HTML(;
    #     prettyurls = get(ENV, "CI", "") == "true",
    #     assets = String[],
    #     mathengine,
    #     edit_link = nothing,
    #     size_threshold = nothing,
    # ),
    format=DocumenterVitepress.MarkdownVitepress(
        repo = "https://github.com/ashutosh-b-b/d2l-julia", devbranch = "master", devurl = "dev"),

    repo = "https://github.com/ashutosh-b-b/d2l-julia",
    pagesonly = true,
    pages = ["Home" => "index.md"; 
        "Chapters" => [
            "chapters.md";
            pages[1:8]
        ];
        "References" => "references.md"
    ],
    # pages = pages,
    build = joinpath(@__DIR__, "build"),
    warnonly = [:missing_docs],
    linkcheck = false,
    doctest = false,
    clean = true,
    plugins = [bib]
)

# makedocs(;
#     authors = "Ashutosh Bharambe",
#     sitename = "d2l Julia",
#     remotes = nothing,
#     format = Documenter.HTML(;
#         prettyurls = get(ENV, "CI", nothing) == "true",
#         assets = String[],
#         mathengine,
#         edit_link = nothing),
#     pagesonly = true,
#     doctest = true,
#     linkcheck = true,
#     warnonly = [:missing_docs],
#     # pages = [
#     #     "5. Convolutional Neural Networks" => [
#     #         "CH5.Convolutional_Neural_Networks/CNN_2.md",
#     #         "CH5.Convolutional_Neural_Networks/CNN_3.md",
#     #         "CH5.Convolutional_Neural_Networks/CNN_4.md",
#     #         "CH5.Convolutional_Neural_Networks/CNN_5.md",
#     #         "CH5.Convolutional_Neural_Networks/CNN_6.md",
#     #     ],
#     #     "6. Modern Convolutional Neural Networks" => [
#     #         "CH6.ModernConvolutionalNeuralNetworks/MCNN_0.md",
#     #         "CH6.ModernConvolutionalNeuralNetworks/MCNN_1.md",
#     #         "CH6.ModernConvolutionalNeuralNetworks/MCNN_2.md",
#     #         "CH6.ModernConvolutionalNeuralNetworks/MCNN_3.md",
#     #         "CH6.ModernConvolutionalNeuralNetworks/MCNN_4.md",
#     #         "CH6.ModernConvolutionalNeuralNetworks/MCNN_5.md",
#     #         "CH6.ModernConvolutionalNeuralNetworks/MCNN_6.md",
#     #         "CH6.ModernConvolutionalNeuralNetworks/MCNN_7.md",
#     #         "CH6.ModernConvolutionalNeuralNetworks/MCNN_8.md",
#     #     ],
#     #     "7. Recurrent Neural Networks" => [
#     #         "CH7.Recurrent_Neural_Networks/RNN_0.md",
#     #         "CH7.Recurrent_Neural_Networks/RNN_1.md",
#     #         "CH7.Recurrent_Neural_Networks/RNN_2.md",
#     #         "CH7.Recurrent_Neural_Networks/RNN_3.md",
#     #         "CH7.Recurrent_Neural_Networks/RNN_4.md",
#     #         "CH7.Recurrent_Neural_Networks/RNN_5.md",
#     #         "CH7.Recurrent_Neural_Networks/RNN_6.md",
#     #         "CH7.Recurrent_Neural_Networks/RNN_7.md",
#     #     ],
#     #     "8. Modern Recurrent Neural Networks" => [
#     #         "CH8.Modern_Recurrent_Neural_Networks/MRNN_1.md",
#     #         "CH8.Modern_Recurrent_Neural_Networks/MRNN_2.md",
#     #         "CH8.Modern_Recurrent_Neural_Networks/MRNN_3.md",
#     #         "CH8.Modern_Recurrent_Neural_Networks/MRNN_4.md",
#     #         "CH8.Modern_Recurrent_Neural_Networks/MRNN_5.md",
#     #         "CH8.Modern_Recurrent_Neural_Networks/MRNN_6.md",
#     #         "CH8.Modern_Recurrent_Neural_Networks/MRNN7.md",
#     #     ],
#     #     "9. Attention Mechanisms and Transformers" => [
#     #         "CH9.Attention_Mechanisms_and_Transformers/ATTN_1.md",
#     #         "CH9.Attention_Mechanisms_and_Transformers/ATTN_2.md",
#     #         "CH9.Attention_Mechanisms_and_Transformers/ATTN_3.md",
#     #         "CH9.Attention_Mechanisms_and_Transformers/ATTN_4.md",
#     #         "CH9.Attention_Mechanisms_and_Transformers/ATTN_5.md",
#     #         "CH9.Attention_Mechanisms_and_Transformers/ATTN_6.md",
#     #     ],]
#     pages = pages,
#     )

DocumenterVitepress.deploydocs(;
    repo = "github.com/ashutosh-b-b/d2l-julia",
    branch = "gh-pages",
    push_preview = true)