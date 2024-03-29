struct VGG19
    conv1_1_w
    conv1_1_b
    conv1_2_w
    conv1_2_b
    conv2_1_w
    conv2_1_b
    conv2_2_w
    conv2_2_b
    conv3_1_w
    conv3_1_b
    conv3_2_w
    conv3_2_b
    conv3_3_w
    conv3_3_b
    conv3_4_w
    conv3_4_b
    conv4_1_w
    conv4_1_b
    conv4_2_w
    conv4_2_b
    conv4_3_w
    conv4_3_b
    conv4_4_w
    conv4_4_b
    conv5_1_w
    conv5_1_b
    conv5_2_w
    conv5_2_b
    conv5_3_w
    conv5_3_b
    conv5_4_w
    conv5_4_b
    fc6_w
    fc6_b
    fc7_w
    fc7_b
    fc8_w
    fc8_b
    mean_color
    class_names
    class_descs
end

struct VGG19Iterator
    nn::VGG19
    x0
end

function (nn::VGG19)(x)::VGG19Iterator
    VGG19Iterator(nn, x)
end

function Base.iterate(it::VGG19Iterator)
    Base.iterate(it, (it.x0, :conv1_1))
end

function Base.iterate(it::VGG19Iterator, state::Tuple{Any, Symbol})
    # decompose state
    (x, layer) = state
    # compute
    if layer == :conv1_1
        x = conv4(it.nn.conv1_1_w, x; padding=1, stride=1, mode=0) .+ it.nn.conv1_1_b
        x = relu.(x)
        return ((:conv1_1, x), (x, :conv1_2))
    elseif layer == :conv1_2
        x = conv4(it.nn.conv1_2_w, x; padding=1, stride=1, mode=0) .+ it.nn.conv1_2_b
        x = relu.(x)
        return ((:conv1_2, x), (x, :pool1))
    elseif layer == :pool1
        x = pool(x; window=2, padding=0, stride=2, mode=0)
        return ((:pool1, x), (x, :conv2_1))
    elseif layer == :conv2_1
        x = conv4(it.nn.conv2_1_w, x; padding=1, stride=1, mode=0) .+ it.nn.conv2_1_b
        x = relu.(x)
        return ((:conv2_1, x), (x, :conv2_2))
    elseif layer == :conv2_2
        x = conv4(it.nn.conv2_2_w, x; padding=1, stride=1, mode=0) .+ it.nn.conv2_2_b
        x = relu.(x)
        return ((:conv2_2, x), (x, :pool2))
    elseif layer == :pool2
        x = pool(x; window=2, padding=0, stride=2, mode=0)
        return ((:pool2, x), (x, :conv3_1))
    elseif layer == :conv3_1
        x = conv4(it.nn.conv3_1_w, x; padding=1, stride=1, mode=0) .+ it.nn.conv3_1_b
        x = relu.(x)
        return ((:conv3_1, x), (x, :conv3_2))
    elseif layer == :conv3_2
        x = conv4(it.nn.conv3_2_w, x; padding=1, stride=1, mode=0) .+ it.nn.conv3_2_b
        x = relu.(x)
        return ((:conv3_2, x), (x, :conv3_3))
    elseif layer == :conv3_3
        x = conv4(it.nn.conv3_3_w, x; padding=1, stride=1, mode=0) .+ it.nn.conv3_3_b
        x = relu.(x)
        return ((:conv3_3, x), (x, :conv3_4))
    elseif layer == :conv3_4
        x = conv4(it.nn.conv3_4_w, x; padding=1, stride=1, mode=0) .+ it.nn.conv3_4_b
        x = relu.(x)
        return ((:conv3_4, x), (x, :pool3))
    elseif layer == :pool3
        x = pool(x; window=2, padding=0, stride=2, mode=0)
        return ((:pool3, x), (x, :conv4_1))
    elseif layer == :conv4_1
        x = conv4(it.nn.conv4_1_w, x; padding=1, stride=1, mode=0) .+ it.nn.conv4_1_b
        x = relu.(x)
        return ((:conv4_1, x), (x, :conv4_2))
    elseif layer == :conv4_2
        x = conv4(it.nn.conv4_2_w, x; padding=1, stride=1, mode=0) .+ it.nn.conv4_2_b
        x = relu.(x)
        return ((:conv4_2, x), (x, :conv4_3))
    elseif layer == :conv4_3
        x = conv4(it.nn.conv4_3_w, x; padding=1, stride=1, mode=0) .+ it.nn.conv4_3_b
        x = relu.(x)
        return ((:conv4_3, x), (x, :conv4_4))
    elseif layer == :conv4_4
        x = conv4(it.nn.conv4_4_w, x; padding=1, stride=1, mode=0) .+ it.nn.conv4_4_b
        x = relu.(x)
        return ((:conv4_4, x), (x, :pool4))
    elseif layer == :pool4
        x = pool(x; window=2, padding=0, stride=2, mode=0)
        return ((:pool4, x), (x, :conv5_1))
    elseif layer == :conv5_1
        x = conv4(it.nn.conv5_1_w, x; padding=1, stride=1, mode=0) .+ it.nn.conv5_1_b
        x = relu.(x)
        return ((:conv5_1, x), (x, :conv5_2))
    elseif layer == :conv5_2
        x = conv4(it.nn.conv5_2_w, x; padding=1, stride=1, mode=0) .+ it.nn.conv5_2_b
        x = relu.(x)
        return ((:conv5_2, x), (x, :conv5_3))
    elseif layer == :conv5_3
        x = conv4(it.nn.conv5_3_w, x; padding=1, stride=1, mode=0) .+ it.nn.conv5_3_b
        x = relu.(x)
        return ((:conv5_3, x), (x, :conv5_4))
    elseif layer == :conv5_4
        x = conv4(it.nn.conv5_4_w, x; padding=1, stride=1, mode=0) .+ it.nn.conv5_4_b
        x = relu.(x)
        return ((:conv5_4, x), (x, :pool5))
    elseif layer == :pool5
        x = pool(x; window=2, padding=0, stride=2, mode=0)
        return ((:pool5, x), (x, :fc6))
    elseif layer == :fc6
        x = conv4(it.nn.fc6_w, x; padding=0, stride=1, mode=0) .+ it.nn.fc6_b
        x = relu.(x)
        return ((:fc6, x), (x, :fc7))
    elseif layer == :fc7
        x = conv4(it.nn.fc7_w, x; padding=0, stride=1, mode=0) .+ it.nn.fc7_b
        x = relu.(x)
        return ((:fc7, x), (x, :fc8))
    elseif layer == :fc8
        n = size(it.x0, 4)
        x = conv4(it.nn.fc8_w, x; padding=0, stride=1, mode=0) .+ it.nn.fc8_b
        x = reshape(x,:,n)
        return ((:fc8, x), (x, :prob))
    elseif layer == :prob
        x = softmax(x)
        return ((:prob, x), nothing)
    else
        throw(InvalidStateException("Invalid Layer", layer))
    end
end

Base.iterate(it::VGG19Iterator, ::Nothing) = nothing
Base.IteratorSize(it::VGG19Iterator) = Base.HasLength()
Base.length(it::VGG19Iterator) = 25

function load_model(::Type{VGG19}; atype=default_array_type())::VGG19
    arr(a) = convert(atype, a)
    fcw(a) = permutedims(a,(2,1,3,4)) |> arr
    fcb(a) = reshape(a,1,1,:) |> arr
    #
    m = read_vgg19()
    @assert m["layers"][ 1]["name"] == "conv1_1"
    conv1_1_w, conv1_1_b = m["layers"][1]["weights"]
    conv1_1_w = conv1_1_w |> fcw
    conv1_1_b = conv1_1_b |> fcb
    @assert m["layers"][ 2]["name"] == "relu1_1"
    @assert m["layers"][ 3]["name"] == "conv1_2"
    conv1_2_w, conv1_2_b = m["layers"][3]["weights"]
    conv1_2_w = conv1_2_w |> fcw
    conv1_2_b = conv1_2_b |> fcb
    @assert m["layers"][ 4]["name"] == "relu1_2"
    @assert m["layers"][ 5]["name"] == "pool1"
    @assert m["layers"][ 6]["name"] == "conv2_1"
    conv2_1_w, conv2_1_b = m["layers"][6]["weights"]
    conv2_1_w = conv2_1_w |> fcw
    conv2_1_b = conv2_1_b |> fcb
    @assert m["layers"][ 7]["name"] == "relu2_1"
    @assert m["layers"][ 8]["name"] == "conv2_2"
    conv2_2_w, conv2_2_b = m["layers"][8]["weights"]
    conv2_2_w = conv2_2_w |> fcw
    conv2_2_b = conv2_2_b |> fcb
    @assert m["layers"][ 9]["name"] == "relu2_2"
    @assert m["layers"][10]["name"] == "pool2"
    @assert m["layers"][11]["name"] == "conv3_1"
    conv3_1_w, conv3_1_b = m["layers"][11]["weights"]
    conv3_1_w = conv3_1_w |> fcw
    conv3_1_b = conv3_1_b |> fcb
    @assert m["layers"][12]["name"] == "relu3_1"
    @assert m["layers"][13]["name"] == "conv3_2"
    conv3_2_w, conv3_2_b = m["layers"][13]["weights"]
    conv3_2_w = conv3_2_w |> fcw
    conv3_2_b = conv3_2_b |> fcb
    @assert m["layers"][14]["name"] == "relu3_2"
    @assert m["layers"][15]["name"] == "conv3_3"
    conv3_3_w, conv3_3_b = m["layers"][15]["weights"]
    conv3_3_w = conv3_3_w |> fcw
    conv3_3_b = conv3_3_b |> fcb
    @assert m["layers"][16]["name"] == "relu3_3"
    @assert m["layers"][17]["name"] == "conv3_4"
    conv3_4_w, conv3_4_b = m["layers"][17]["weights"]
    conv3_4_w = conv3_4_w |> fcw
    conv3_4_b = conv3_4_b |> fcb
    @assert m["layers"][18]["name"] == "relu3_4"
    @assert m["layers"][19]["name"] == "pool3"
    @assert m["layers"][20]["name"] == "conv4_1"
    conv4_1_w, conv4_1_b = m["layers"][20]["weights"]
    conv4_1_w = conv4_1_w |> fcw
    conv4_1_b = conv4_1_b |> fcb
    @assert m["layers"][21]["name"] == "relu4_1"
    @assert m["layers"][22]["name"] == "conv4_2"
    conv4_2_w, conv4_2_b = m["layers"][22]["weights"]
    conv4_2_w = conv4_2_w |> fcw
    conv4_2_b = conv4_2_b |> fcb
    @assert m["layers"][23]["name"] == "relu4_2"
    @assert m["layers"][24]["name"] == "conv4_3"
    conv4_3_w, conv4_3_b = m["layers"][24]["weights"]
    conv4_3_w = conv4_3_w |> fcw
    conv4_3_b = conv4_3_b |> fcb
    @assert m["layers"][25]["name"] == "relu4_3"
    @assert m["layers"][26]["name"] == "conv4_4"
    conv4_4_w, conv4_4_b = m["layers"][26]["weights"]
    conv4_4_w = conv4_4_w |> fcw
    conv4_4_b = conv4_4_b |> fcb
    @assert m["layers"][27]["name"] == "relu4_4"
    @assert m["layers"][28]["name"] == "pool4"
    @assert m["layers"][29]["name"] == "conv5_1"
    conv5_1_w, conv5_1_b = m["layers"][29]["weights"]
    conv5_1_w = conv5_1_w |> fcw
    conv5_1_b = conv5_1_b |> fcb
    @assert m["layers"][30]["name"] == "relu5_1"
    @assert m["layers"][31]["name"] == "conv5_2"
    conv5_2_w, conv5_2_b = m["layers"][31]["weights"]
    conv5_2_w = conv5_2_w |> fcw
    conv5_2_b = conv5_2_b |> fcb
    @assert m["layers"][32]["name"] == "relu5_2"
    @assert m["layers"][33]["name"] == "conv5_3"
    conv5_3_w, conv5_3_b = m["layers"][33]["weights"]
    conv5_3_w = conv5_3_w |> fcw
    conv5_3_b = conv5_3_b |> fcb
    @assert m["layers"][34]["name"] == "relu5_3"
    @assert m["layers"][35]["name"] == "conv5_4"
    conv5_4_w, conv5_4_b = m["layers"][35]["weights"]
    conv5_4_w = conv5_4_w |> fcw
    conv5_4_b = conv5_4_b |> fcb
    @assert m["layers"][36]["name"] == "relu5_4"
    @assert m["layers"][37]["name"] == "pool5"
    @assert m["layers"][38]["name"] == "fc6"
    fc6_w, fc6_b = m["layers"][38]["weights"]
    fc6_w = fc6_w |> fcw
    fc6_b = fc6_b |> fcb
    @assert m["layers"][39]["name"] == "relu6"
    @assert m["layers"][40]["name"] == "fc7"
    fc7_w, fc7_b = m["layers"][40]["weights"]
    fc7_w = fc7_w |> fcw
    fc7_b = fc7_b |> fcb
    @assert m["layers"][41]["name"] == "relu7"
    @assert m["layers"][42]["name"] == "fc8"
    fc8_w, fc8_b = m["layers"][42]["weights"]
    fc8_w = fc8_w |> fcw
    fc8_b = fc8_b |> fcb
    @assert m["layers"][43]["name"] == "prob"
    #
    model = VGG19(
        conv1_1_w, conv1_1_b,
        conv1_2_w, conv1_2_b,
        conv2_1_w, conv2_1_b,
        conv2_2_w, conv2_2_b,
        conv3_1_w, conv3_1_b,
        conv3_2_w, conv3_2_b,
        conv3_3_w, conv3_3_b,
        conv3_4_w, conv3_4_b,
        conv4_1_w, conv4_1_b,
        conv4_2_w, conv4_2_b,
        conv4_3_w, conv4_3_b,
        conv4_4_w, conv4_4_b,
        conv5_1_w, conv5_1_b,
        conv5_2_w, conv5_2_b,
        conv5_3_w, conv5_3_b,
        conv5_4_w, conv5_4_b,
        fc6_w, fc6_b,
        fc7_w, fc7_b,
        fc8_w, fc8_b,
        # mean_color
        m["meta"]["normalization"]["averageImage"] .|> Float32,
        # classes
        m["meta"]["classes"]["name"],
        m["meta"]["classes"]["description"],
    )
    #
    model
end
