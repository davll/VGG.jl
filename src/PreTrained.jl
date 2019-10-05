module PreTrained

using DataDeps
using MAT

const DEPNAME = "VGG.ImageNet.VeryDeep"
const BASE_URL = "http://www.vlfeat.org/matconvnet/models/"
const VGG16_FILENAME = "imagenet-vgg-verydeep-16.mat"
const VGG19_FILENAME = "imagenet-vgg-verydeep-19.mat"

function read_data(filename)
    dir = @datadep_str DEPNAME
    path = joinpath(dir, filename)
    matread(path)
end

read_vgg16() = read_data(VGG16_FILENAME)
read_vgg19() = read_data(VGG19_FILENAME)

function __init__()
    register(DataDep(
        DEPNAME,
        """
        Model: ImageNet-VGG-VeryDeep
        Authors: Karen Simonyan and Andrew Zisserman
        Website: http://www.robots.ox.ac.uk/~vgg/research/very_deep/
        
        [Karen Simonyan and Andrew Zisserman, 2014]
            Karen Simonyan and Andrew Zisserman.
            "Very Deep Convolutional Networks for Large-Scale Image Recognition"
            arXiv technical report, 2014
            https://arxiv.org/abs/1409.1556
        
        @misc{simonyan2014deep,
            title={Very Deep Convolutional Networks for Large-Scale Image Recognition},
            author={Karen Simonyan and Andrew Zisserman},
            year={2014},
            eprint={1409.1556},
            archivePrefix={arXiv},
            primaryClass={cs.CV}
        }
        """,
        BASE_URL .* [VGG16_FILENAME, VGG19_FILENAME],
    ))
end

end
