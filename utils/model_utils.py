RESNET_MAPPING = {
    "layer1.0": "body.0",
    "layer1.1": "body.1",
    "layer1.2": "body.2",
    "layer2.0": "body.3",
    "layer2.1": "body.4",
    "layer2.2": "body.5",
    "layer2.3": "body.6",
    "layer3.0": "body.7",
    "layer3.1": "body.8",
    "layer3.2": "body.9",
    "layer3.3": "body.10",
    "layer3.4": "body.11",
    "layer3.5": "body.12",
    "layer4.0": "body.13",
    "layer4.1": "body.14",
    "layer4.2": "body.15",
}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def toogle_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def stylegan_to_classifier(x, out_size=(224, 224)):
    """Clip image to range(0,1)"""
    img_tmp = x.clone()
    img_tmp = torch.clamp((0.5*img_tmp + 0.5), 0, 1)
    img_tmp = F.interpolate(img_tmp, size=out_size, mode='bilinear')
    img_tmp[:,0] = (img_tmp[:,0] - 0.485)/0.229
    img_tmp[:,1] = (img_tmp[:,1] - 0.456)/0.224
    img_tmp[:,2] = (img_tmp[:,2] - 0.406)/0.225
    return img_tmp


def get_stylespace_from_w(w, G):
    style_space = []
    to_rgb_stylespaces = []

    noise = [getattr(G.noises, 'noise_{}'.format(i)) for i in range(G.num_layers)]
    latent = w
    style_space.append(G.conv1.conv.modulation(latent[:, 0]))
    to_rgb_stylespaces.append(G.to_rgb1.conv.modulation(latent[:, 1]))

    i = 1
    for conv1, conv2, noise1, noise2, to_rgb in zip(
            G.convs[::2], G.convs[1::2], noise[1::2], noise[2::2], G.to_rgbs
    ):
        style_space.append(conv1.conv.modulation(latent[:, i]))
        style_space.append(conv2.conv.modulation(latent[:, i + 1]))
        to_rgb_stylespaces.append(to_rgb.conv.modulation(latent[:, i + 2]))
        i += 2
    return style_space, to_rgb_stylespaces


def get_stylespace_from_w_hyperinv(w, G):
    with torch.no_grad():
        style_space = []
        to_rgb_stylespaces = []
        G = G.synthesis

        block_ws = []
        w_idx = 0
        for res in G.block_resolutions:
            block = getattr(G, f"b{res}")
            block_ws.append(w.narrow(1, w_idx, block.num_conv + block.num_torgb))
            w_idx += block.num_conv

        i = 0
        for res, cur_ws in zip(G.block_resolutions, block_ws):
            block = getattr(G, f"b{res}")
            if i != 0:
                style_space.append(block.conv0.affine(w[:, i]))
                i += 1
            style_space.append(block.conv1.affine(w[:, i]))
            i += 1
            to_rgb_stylespaces.append(block.torgb.affine(w[:, i]))

    return style_space, to_rgb_stylespaces
