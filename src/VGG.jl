"""
VGG's convolutional neural networks

Publication:
    Very Deep Convolutional Networks for Large-Scale Image Recognition
    https://arxiv.org/abs/1409.1556

"""
module VGG

export load_model, save_model, fit, loss
export VGG19, VGG16
export preprocess_image
export postprocess_image

include("PreTrained.jl")

import .PreTrained: read_vgg16, read_vgg19
using Knet
using ImageCore

function load_model end
function save_model end
function fit end
function loss end

function default_array_type()
    if Knet.gpu() >= 0
        KnetArray{Float32}
    else
        Array{Float32}
    end
end

include("vgg16.jl")
include("vgg19.jl")

function preprocess_image(vgg::Union{VGG16, VGG19}, img)
    x = channelview(img) .|> Float32
    x = permutedims(x, (2,3,1))
    x = x .* 255 .- vgg.mean_color
    x = reshape(x, size(x)..., 1)
    return convert(typeof(vgg.conv1_1_w), x)
end

function postprocess_image(vgg::Union{VGG16, VGG19}, x; clamp_pixels=true)
    x = value(x)
    x = convert(Array{Float32,4}, x)
    x = reshape(x, size(x)[1:end-1]...)
    x = (x .+ vgg.mean_color) ./ 255
    if clamp_pixels
        x = clamp.(x, 0.0f0, 1.0f0)
    end
    x = permutedims(x, (3,1,2))
    return colorview(RGB, x)
end

end
