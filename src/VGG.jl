"""
VGG's convolutional neural networks

Publication:
    Very Deep Convolutional Networks for Large-Scale Image Recognition
    https://arxiv.org/abs/1409.1556
"""
module VGG

export load_model, save_model, fit, loss
export VGG19, VGG16

include("PreTrained.jl")

import .PreTrained: read_vgg16, read_vgg19
using Knet

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

end
