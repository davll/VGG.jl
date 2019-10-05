struct VGG16
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
    conv4_1_w
    conv4_1_b
    conv4_2_w
    conv4_2_b
    conv4_3_w
    conv4_3_b
    conv5_1_w
    conv5_1_b
    conv5_2_w
    conv5_2_b
    conv5_3_w
    conv5_3_b
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

function (nn::VGG16)(x)
    # check input
    @assert ndims(x) == 4
    w, h, d, n = size(x)
    @assert w == 224
    @assert h == 224
    @assert d == 3
    # feed forward
    # conv1_1
    x = conv4(nn.conv1_1_w, x; padding=1, stride=1, mode=0) .+ nn.conv1_1_b
    x = relu.(x)
    @assert size(x) == (224,224,64,n)
    # conv1_2
    x = conv4(nn.conv1_2_w, x; padding=1, stride=1, mode=0) .+ nn.conv1_2_b
    x = relu.(x)
    @assert size(x) == (224,224,64,n)
    # pool1
    x = pool(x; window=2, padding=0, stride=2, mode=0)
    @assert size(x) == (112,112,64,n)
    # conv2_1
    x = conv4(nn.conv2_1_w, x; padding=1, stride=1, mode=0) .+ nn.conv2_1_b
    x = relu.(x)
    @assert size(x) == (112,112,128,n)
    # conv2_2
    x = conv4(nn.conv2_2_w, x; padding=1, stride=1, mode=0) .+ nn.conv2_2_b
    x = relu.(x)
    @assert size(x) == (112,112,128,n)
    # pool2
    x = pool(x; window=2, padding=0, stride=2, mode=0)
    @assert size(x) == (56,56,128,n)
    # conv3_1
    x = conv4(nn.conv3_1_w, x; padding=1, stride=1, mode=0) .+ nn.conv3_1_b
    x = relu.(x)
    @assert size(x) == (56,56,256,n)
    # conv3_2
    x = conv4(nn.conv3_2_w, x; padding=1, stride=1, mode=0) .+ nn.conv3_2_b
    x = relu.(x)
    @assert size(x) == (56,56,256,n)
    # conv3_3
    x = conv4(nn.conv3_3_w, x; padding=1, stride=1, mode=0) .+ nn.conv3_3_b
    x = relu.(x)
    @assert size(x) == (56,56,256,n)
    # pool3
    x = pool(x; window=2, padding=0, stride=2, mode=0)
    @assert size(x) == (28,28,256,n)
    # conv4_1
    x = conv4(nn.conv4_1_w, x; padding=1, stride=1, mode=0) .+ nn.conv4_1_b
    x = relu.(x)
    @assert size(x) == (28,28,512,n)
    # conv4_2
    x = conv4(nn.conv4_2_w, x; padding=1, stride=1, mode=0) .+ nn.conv4_2_b
    x = relu.(x)
    @assert size(x) == (28,28,512,n)
    # conv4_3
    x = conv4(nn.conv4_3_w, x; padding=1, stride=1, mode=0) .+ nn.conv4_3_b
    x = relu.(x)
    @assert size(x) == (28,28,512,n)
    # pool4
    x = pool(x; window=2, padding=0, stride=2, mode=0)
    @assert size(x) == (14,14,512,n)
    # conv5_1
    x = conv4(nn.conv5_1_w, x; padding=1, stride=1, mode=0) .+ nn.conv5_1_b
    x = relu.(x)
    @assert size(x) == (14,14,512,n)
    # conv5_2
    x = conv4(nn.conv5_2_w, x; padding=1, stride=1, mode=0) .+ nn.conv5_2_b
    x = relu.(x)
    @assert size(x) == (14,14,512,n)
    # conv5_3
    x = conv4(nn.conv5_3_w, x; padding=1, stride=1, mode=0) .+ nn.conv5_3_b
    x = relu.(x)
    @assert size(x) == (14,14,512,n)
    # pool5
    x = pool(x; window=2, padding=0, stride=2, mode=0)
    @assert size(x) == (7,7,512,n)
    # fc6
    x = conv4(nn.fc6_w, x; padding=0, stride=1, mode=0) .+ nn.fc6_b
    x = relu.(x)
    @assert size(x) == (1,1,4096,n)
    # drop6
    #x = dropout(x, 0.5; drop=training)
    # fc7
    x = conv4(nn.fc7_w, x; padding=0, stride=1, mode=0) .+ nn.fc7_b
    x = relu.(x)
    @assert size(x) == (1,1,4096,n)
    # drop7
    #x = dropout(x, 0.5; drop=training)
    # fc8
    x = conv4(nn.fc8_w, x; padding=0, stride=1, mode=0) .+ nn.fc8_b
    @assert size(x) == (1,1,1000,n)
    x = reshape(x,:,n)
    @assert size(x) == (1000,n)
    # prob
    softmax(x)
end

function load_model(::Type{VGG16}; atype=default_array_type())::VGG16
    arr(a) = convert(atype, a)
    fcw(a) = permutedims(a,(2,1,3,4)) |> arr
    fcb(a) = reshape(a,1,1,:) |> arr
    #
    m = read_vgg16()
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
    @assert m["layers"][17]["name"] == "pool3"
    @assert m["layers"][18]["name"] == "conv4_1"
    conv4_1_w, conv4_1_b = m["layers"][18]["weights"]
    conv4_1_w = conv4_1_w |> fcw
    conv4_1_b = conv4_1_b |> fcb
    @assert m["layers"][19]["name"] == "relu4_1"
    @assert m["layers"][20]["name"] == "conv4_2"
    conv4_2_w, conv4_2_b = m["layers"][20]["weights"]
    conv4_2_w = conv4_2_w |> fcw
    conv4_2_b = conv4_2_b |> fcb
    @assert m["layers"][21]["name"] == "relu4_2"
    @assert m["layers"][22]["name"] == "conv4_3"
    conv4_3_w, conv4_3_b = m["layers"][22]["weights"]
    conv4_3_w = conv4_3_w |> fcw
    conv4_3_b = conv4_3_b |> fcb
    @assert m["layers"][23]["name"] == "relu4_3"
    @assert m["layers"][24]["name"] == "pool4"
    @assert m["layers"][25]["name"] == "conv5_1"
    conv5_1_w, conv5_1_b = m["layers"][25]["weights"]
    conv5_1_w = conv5_1_w |> fcw
    conv5_1_b = conv5_1_b |> fcb
    @assert m["layers"][26]["name"] == "relu5_1"
    @assert m["layers"][27]["name"] == "conv5_2"
    conv5_2_w, conv5_2_b = m["layers"][27]["weights"]
    conv5_2_w = conv5_2_w |> fcw
    conv5_2_b = conv5_2_b |> fcb
    @assert m["layers"][28]["name"] == "relu5_2"
    @assert m["layers"][29]["name"] == "conv5_3"
    conv5_3_w, conv5_3_b = m["layers"][29]["weights"]
    conv5_3_w = conv5_3_w |> fcw
    conv5_3_b = conv5_3_b |> fcb
    @assert m["layers"][30]["name"] == "relu5_3"
    @assert m["layers"][31]["name"] == "pool5"
    @assert m["layers"][32]["name"] == "fc6"
    fc6_w, fc6_b = m["layers"][32]["weights"]
    fc6_w = fc6_w |> fcw
    fc6_b = fc6_b |> fcb
    @assert m["layers"][33]["name"] == "relu6"
    @assert m["layers"][34]["name"] == "fc7"
    fc7_w, fc7_b = m["layers"][34]["weights"]
    fc7_w = fc7_w |> fcw
    fc7_b = fc7_b |> fcb
    @assert m["layers"][35]["name"] == "relu7"
    @assert m["layers"][36]["name"] == "fc8"
    fc8_w, fc8_b = m["layers"][36]["weights"]
    fc8_w = fc8_w |> fcw
    fc8_b = fc8_b |> fcb
    @assert m["layers"][37]["name"] == "prob"
    #
    model = VGG16(
        conv1_1_w, conv1_1_b,
        conv1_2_w, conv1_2_b,
        conv2_1_w, conv2_1_b,
        conv2_2_w, conv2_2_b,
        conv3_1_w, conv3_1_b,
        conv3_2_w, conv3_2_b,
        conv3_3_w, conv3_3_b,
        conv4_1_w, conv4_1_b,
        conv4_2_w, conv4_2_b,
        conv4_3_w, conv4_3_b,
        conv5_1_w, conv5_1_b,
        conv5_2_w, conv5_2_b,
        conv5_3_w, conv5_3_b,
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
