# coding=utf-8

import copy
import logging
import math
from os.path import join as pjoin

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from scipy import ndimage

from medicalseg.cvlibs import manager
from medicalseg.models.backbone.vit_seg_modeling_resnet_skip import ResNetV2
from medicalseg.models.param_init import xavier_normal_
from medicalseg.utils import load_pretrained_model

logger = logging.getLogger(__name__)

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return paddle.to_tensor(weights)


def swish(x):
    return x * F.sigmoid(x)


ACT2FN = {"gelu": F.gelu, "relu": F.relu, "swish": swish}


class Attention(nn.Layer):
    def __init__(self, hidden_size, num_heads, attention_dropout_rate, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = num_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(attention_dropout_rate)
        self.proj_dropout = nn.Dropout(attention_dropout_rate)

        self.softmax = nn.Softmax(axis=-1)

    def transpose_for_scores(self, x):
        new_x_shape = list(x.shape[:-1]) + [
            self.num_attention_heads, self.attention_head_size
        ]
        x = x.reshape(new_x_shape)
        return x.transpose([0, 2, 1, 3])

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = paddle.matmul(query_layer,
                                         key_layer.transpose([0, 1, 3, 2]))
        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = paddle.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose([0, 2, 1, 3])
        new_context_layer_shape = list(
            context_layer.shape[:-2]) + [self.all_head_size]
        context_layer = context_layer.reshape(new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Layer):
    def __init__(self, hidden_size, mlp_dim, dropout_rate):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(hidden_size, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = nn.Dropout(dropout_rate)

        self._init_weights()

    def _init_weights(self):
        xavier_normal_(self.fc1.weight)
        xavier_normal_(self.fc2.weight)
        nn.initializer.Normal(mean=0, std=1e-6)(self.fc1.bias)
        nn.initializer.Normal(mean=0, std=1e-6)(self.fc2.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Layer):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, grid_size, resnet_num_layers, width_factor, hidden_size,
                 dropout_rate, img_size):
        super(Embeddings, self).__init__()

        img_size = (img_size, img_size)

        patch_size = (img_size[0] // 16 // grid_size[0],
                      img_size[1] // 16 // grid_size[1])
        patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
        n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] //
                                                           patch_size_real[1])

        self.hybrid_model = ResNetV2(
            block_units=resnet_num_layers, width_factor=width_factor)
        in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = nn.Conv2D(
            in_channels=in_channels,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size)
        self.position_embeddings = paddle.create_parameter(
            shape=[1, n_patches, hidden_size],
            dtype='float32',
            default_initializer=nn.initializer.Constant(0))

        self.dropout = nn.Dropout(dropout_rate)
        self.hybrid = True

    def forward(self, x):
        x, features = self.hybrid_model(x)
        x = self.patch_embeddings(
            x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = paddle.transpose(x, [0, 2, 1])

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


class Block(nn.Layer):
    def __init__(self, hidden_size, mlp_dim, dropout_rate, num_heads,
                 attention_dropout_rate, vis):
        super(Block, self).__init__()
        self.hidden_size = hidden_size
        self.attention_norm = nn.LayerNorm(hidden_size, epsilon=1e-6)
        self.ffn_norm = nn.LayerNorm(hidden_size, epsilon=1e-6)
        self.ffn = Mlp(hidden_size, mlp_dim, dropout_rate)
        self.attn = Attention(hidden_size, num_heads, attention_dropout_rate,
                              vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with paddle.no_grad():
            query_weight = np2th(weights[pjoin(
                ROOT, ATTENTION_Q, "kernel")]).reshape(
                    [self.hidden_size, self.hidden_size])
            key_weight = np2th(weights[pjoin(
                ROOT, ATTENTION_K, "kernel")]).reshape(
                    [self.hidden_size, self.hidden_size])
            value_weight = np2th(weights[pjoin(
                ROOT, ATTENTION_V, "kernel")]).reshape(
                    [self.hidden_size, self.hidden_size])
            out_weight = np2th(weights[pjoin(
                ROOT, ATTENTION_OUT, "kernel")]).reshape(
                    [self.hidden_size, self.hidden_size])

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q,
                                             "bias")]).reshape([-1])
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).reshape(
                [-1])
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V,
                                             "bias")]).reshape([-1])
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT,
                                           "bias")]).reshape([-1])

            self.attn.query.weight.set_value(query_weight)
            self.attn.key.weight.set_value(key_weight)
            self.attn.value.weight.set_value(value_weight)
            self.attn.out.weight.set_value(out_weight)
            self.attn.query.bias.set_value(query_bias)
            self.attn.key.bias.set_value(key_bias)
            self.attn.value.bias.set_value(value_bias)
            self.attn.out.bias.set_value(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")])
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")])
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).T
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).T

            self.ffn.fc1.weight.set_value(mlp_weight_0)
            self.ffn.fc2.weight.set_value(mlp_weight_1)
            self.ffn.fc1.bias.set_value(mlp_bias_0)
            self.ffn.fc2.bias.set_value(mlp_bias_1)

            self.attention_norm.weight.set_value(
                np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.set_value(
                np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.set_value(
                np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.set_value(
                np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Layer):
    def __init__(self, hidden_size, num_layers, mlp_dim, dropout_rate,
                 num_heads, attention_dropout_rate, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.LayerList()
        self.encoder_norm = nn.LayerNorm(hidden_size, epsilon=1e-6)
        for _ in range(num_layers):
            layer = Block(hidden_size, mlp_dim, dropout_rate, num_heads,
                          attention_dropout_rate, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Layer):
    def __init__(self, grid_size, resnet_num_layers, width_factor, hidden_size,
                 dropout_rate, num_layers, mlp_dim, num_heads,
                 attention_dropout_rate, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(
            grid_size,
            resnet_num_layers,
            width_factor,
            hidden_size,
            dropout_rate,
            img_size=img_size)
        self.encoder = Encoder(hidden_size, num_layers, mlp_dim, dropout_rate,
                               num_heads, attention_dropout_rate, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(
            embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True, ):
        conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias_attr=not (use_batchnorm), )
        relu = nn.ReLU()

        bn = nn.BatchNorm2D(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Layer):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True, ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm, )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm, )
        self.up = nn.UpsamplingBilinear2D(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = paddle.concat([x, skip], axis=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2D(
            scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Layer):
    def __init__(self, hidden_size, decoder_channels, n_skip, skip_channels):
        super().__init__()
        head_channels = 512
        self.conv_more = Conv2dReLU(
            hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True, )
        decoder_channels = decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels
        self.n_skip = n_skip
        if n_skip != 0:
            skip_channels = skip_channels
            for i in range(4 - n_skip
                           ):  # re-select the skip channels according to n_skip
                skip_channels[3 - i] = 0

        else:
            skip_channels = [0, 0, 0, 0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch)
            for in_ch, out_ch, sk_ch in zip(in_channels, out_channels,
                                            skip_channels)
        ]
        self.blocks = nn.LayerList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.shape  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.transpose([0, 2, 1])
        x = x.reshape([B, hidden, h, w])
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x


@manager.MODELS.add_component
class VisionTransformer(nn.Layer):
    def __init__(self,
                 activation="softmax",
                 classifier="seg",
                 decoder_channels=[256, 128, 64, 16],
                 hidden_size=768,
                 n_skip=3,
                 patch_size=16,
                 patches_grid=[14, 14],
                 patches_size=[16, 16],
                 pretrained_path=None,
                 representation_size=None,
                 resnet_num_layers=[3, 4, 9],
                 width_factor=1,
                 resnet_pretrained_path=None,
                 skip_channels=[512, 256, 64, 16],
                 attention_dropout_rate=0.0,
                 dropout_rate=0.1,
                 mlp_dim=3072,
                 num_heads=12,
                 num_layers=12,
                 img_size=224,
                 num_classes=9,
                 zero_head=False,
                 vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = classifier
        self.transformer = Transformer(patches_grid, resnet_num_layers,
                                       width_factor, hidden_size, dropout_rate,
                                       num_layers, mlp_dim, num_heads,
                                       attention_dropout_rate, img_size, vis)
        self.decoder = DecoderCup(hidden_size, decoder_channels, n_skip,
                                  skip_channels)
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=num_classes,
            kernel_size=3, )
        if pretrained_path is not None:
            self.load_from(np.load(pretrained_path))

    def forward(self, x):
        x = paddle.squeeze(x, axis=2)
        x = x.tile([1, 3, 1, 1])
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        logits = paddle.unsqueeze(logits, axis=2)
        return [logits]

    def load_from(self, weights):
        with paddle.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.set_value(
                np2th(
                    weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.set_value(
                np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.set_value(
                np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.set_value(
                np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.shape == posemb_new.shape:
                self.transformer.embeddings.position_embeddings.set_value(
                    posemb)
            elif posemb.shape[1] - 1 == posemb_new.shape[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.set_value(
                    posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" %
                            (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.shape[1]
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' %
                      (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape([gs_old, gs_old, -1])
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape([1, gs_new * gs_new, -1])
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.set_value(
                    np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.set_value(
                    np2th(
                        res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).reshape([-1])
                gn_bias = np2th(res_weight["gn_root/bias"]).reshape([-1])
                self.transformer.embeddings.hybrid_model.root.gn.weight.set_value(
                    gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.set_value(
                    gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children(
                ):
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)


def export_weight_names(net):
    print(net.state_dict().keys())
    with open('paddle.txt', 'w') as f:
        for key in net.state_dict().keys():
            f.write(key + '\n')


if __name__ == '__main__':
    from medicalseg.utils import load_pretrained_model
    import paddle
    import numpy as np
    net = VisionTransformer(
        img_size=224, num_classes=9, dropout_rate=0, attention_dropout_rate=0)
    load_pretrained_model(net, '/Users/alex/Desktop/VisionTransformer.pdparams')
    inputs = paddle.to_tensor(
        np.load('/Users/alex/Downloads/project_TransUNet/TransUNet/data.npy'))
    # export_weight_names(net)
    # inputs = paddle.randn([2, 1, 224, 224])
    # net.load_from(
    #     weights=np.load("/Users/alex/Downloads/project_TransUNet/model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz"))
    net.train()

    from medicalseg.models.losses import CrossEntropyLoss, DiceLoss
    from paddle.optimizer import Momentum
    opt = Momentum(
        parameters=net.parameters(),
        learning_rate=0.01,
        momentum=0.9,
        weight_decay=0.0001)
    label = np.load(
        '/Users/alex/Downloads/project_TransUNet/TransUNet/label.npy').astype(
            'int64')
    label = paddle.to_tensor(label)
    label = paddle.unsqueeze(label, axis=1)
    inputs = paddle.unsqueeze(inputs, axis=1)
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss()
    for i in range(10):
        outputs = net(inputs)[0]
        loss_ce = ce_loss(outputs, label)
        loss_dice = dice_loss(outputs, label)[0]
        loss = 0.5 * loss_ce + 0.5 * loss_dice
        loss.backward()

        opt.step()
        net.clear_gradients()

        # print(outputs.mean())
        # print(outputs.std())
        print(loss.numpy()[0])

    pass
