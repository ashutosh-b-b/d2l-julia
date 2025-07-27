# D2L Julia 
<p align="center">
    <img src="d2ljulia-logo.png" alt="Description" width="300" height="300">
</p>

[![Build Status](https://github.com/ashutosh-b-b/d2lai.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ashutosh-b-b/d2lai.jl/actions/workflows/CI.yml?query=branch%3Amain)

A hands-on, interactive deep learning book implemented entirely in [Julia](https://julialang.org), based on the acclaimed [Dive into Deep Learning (D2L.ai)](https://d2l.ai) textbook. Learn modern machine learning techniques with clean, idiomatic Julia code and a strong emphasis on mathematical intuition.

---

## 🚀 Features

- 📚 **Interactive Notebooks** – Every chapter is a live Julia notebook runnable via [Pluto.jl](https://github.com/fonsp/Pluto.jl) or [IJulia.jl](https://julialang.github.io/IJulia.jl/stable/).
- 🔢 **Mathematical Intuition** – Clear derivations and explanations alongside the code.
- 🔍 **Modern Deep Learning** – Covers fundamentals to advanced topics like CNNs, RNNs, Transformers, and beyond.
- 🧪 **Built from Scratch** – Learn by implementing models and algorithms manually before using frameworks.
- 📦 **Flux.jl Powered** – Uses the elegant [Flux.jl](https://fluxml.ai) library for building and training models.
- 🧰 **Modular Utilities** – Reusable utilities for training, data loading, and visualization.

## ✅ Implemented Chapters

- [CH3. Linear Regression](Julia_Notebooks/CH3.Linear_Regression)
- [CH4. Linear Classification](Julia_Notebooks/CH4.Linear_Classification)
- [CH5. MLP](Julia_Notebooks/CH5.MLP)
- [CH6. Convolutional Neural Networks](Julia_Notebooks/CH6.Convolutional_Neural_Networks)
- [CH7. Modern Convolutional Neural Networks](Julia_Notebooks/CH7.ModernConvolutionalNeuralNetworks)
- [CH8. Recurrent Neural Networks](Julia_Notebooks/CH8.Recurrent_Neural_Networks)
- [CH9. Modern Recurrent Neural Networks](Julia_Notebooks/CH9.Modern_Recurrent_Neural_Networks)
- [CH10. Attention Mechanisms and Transformers](Julia_Notebooks/CH10.Attention_Mechanisms_and_Transformers)
- [CH11. Computer Vision](Julia_Notebooks/CH11.Computer_Vision)

## Getting Started
- Clone the repository:

```bash
git clone https://github.com/ashutosh-b-b/d2l-julia.git
```
- Download Julia via [juliaup](https://github.com/JuliaLang/juliaup): 

```bash
curl -fsSL https://install.julialang.org | sh
```
- Install Julia 1.11.4 

```bash
juliaup add 1.11.4
```
- Instantiate and Precompile the d2lai package:

```bash
julia +1.11.4 
```

```julia
julia> using Pkg
julia> Pkg.activate("d2lai/")
julia> Pkg.instantiate()
julia> Pkg.precompile()
```

## 🗺️ Roadmap
- Add full coverage for all D2L chapters
- Add automated testing for notebooks
- Add tests for d2lai package.

## ✍️ Contributing
We welcome contributions!
- Fork this repo
- Open a feature branch
- Submit a pull request (PR)

## 🙌 Acknowledgments
Original book: Dive into Deep Learning by Aston Zhang, Zachary C. Lipton, Mu Li, and Alex J. Smola

Julia libraries: Flux.jl, Zygote.jl, MLDatasets.jl