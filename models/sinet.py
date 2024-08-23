import torch
import torch.nn as nn
import copy
import timm
import torch.nn.functional as F # type: ignore
from einops import rearrange
from models.vit import VisionTransformer, PatchEmbed, Block,resolve_pretrained_cfg, build_model_with_cfg, checkpoint_filter_fn, init_weights_vit_timm

def padding(im, patch_size, fill_value=0):
    # make the image sizes divisible by patch_size
    H, W = im.size(2), im.size(3)
    pad_h, pad_w = 0, 0
    if H % patch_size > 0:
        pad_h = patch_size - (H % patch_size)
    if W % patch_size > 0:
        pad_w = patch_size - (W % patch_size)
    im_padded = im
    if pad_h > 0 or pad_w > 0:
        im_padded = F.pad(im, (0, pad_w, 0, pad_h), value=fill_value)
    return im_padded

def unpadding(y, target_size):
    H, W = target_size
    H_pad, W_pad = y.size(2), y.size(3)
    # crop predictions on extra pixels coming from padding
    extra_h = H_pad - H
    extra_w = W_pad - W
    if extra_h > 0:
        y = y[:, :, :-extra_h]
    if extra_w > 0:
        y = y[:, :, :, :-extra_w]
    return y


class DecoderLinear(nn.Module):
    def __init__(self, n_cls, patch_size, d_encoder):
        super().__init__()

        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_cls = n_cls

        self.head = nn.Linear(self.d_encoder, n_cls)
        self.apply(init_weights_vit_timm)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, x, im_size):
        H, W = im_size
        GS = H // self.patch_size
        x = self.head(x)
        x = x[:,1:, :]
        x = rearrange(x, "b (h w) c -> b c h w", h=GS)

        return x

class ViT_Prompts(VisionTransformer):

    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0., weight_init='', init_values=None,
            embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block):

        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, global_pool=global_pool,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, representation_size=representation_size,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, weight_init=weight_init, init_values=init_values,
            embed_layer=embed_layer, norm_layer=norm_layer, act_layer=act_layer, block_fn=block_fn)


    def forward(self, x, instance_tokens=None, **kwargs):

        x = self.patch_embed(x)


        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        if instance_tokens is not None:
            instance_tokens = instance_tokens.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)

        x = x + self.pos_embed.to(x.dtype)
        # if instance_tokens is not None:
        #     x = torch.cat([x[:,:1,:], instance_tokens, x[:,1:,:]], dim=1)

        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        # if self.global_pool:
        #     x = x[:, 1:].(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        return x



def _create_vision_transformer(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    # NOTE this extra code to support handling of repr size for in21k pretrained models
    pretrained_cfg = resolve_pretrained_cfg(variant)
    print(pretrained_cfg)
    default_num_classes = pretrained_cfg['num_classes']

    # default_num_classes = pretrained_cfg.num_classes
    num_classes = kwargs.get('num_classes', default_num_classes)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        repr_size = None
    # print(pretrained_cfg['url'])
   
    model = build_model_with_cfg(
        ViT_Prompts, variant, pretrained,
        pretrained_cfg=pretrained_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in pretrained_cfg['url'],
        **kwargs)
    
    return model

class SiNet(nn.Module):

    def __init__(self, args):
        super(SiNet, self).__init__()
        self._device = args['device'][0]
        model_kwargs = dict(patch_size=16, embed_dim=args["embd_dim"], depth=12, num_heads=12)
        # a changer pour avoir
        self.image_encoder =_create_vision_transformer(args["variant"], pretrained=True, **model_kwargs)
        self.image_encoder.head = nn.Identity()
        if args["dataset"] == "cddb":
            self.class_num = 2
            # initialisation du pool de classifieurs
            self.classifier_pool = nn.ModuleList([
                nn.Linear(args["embd_dim"], self.class_num, bias=True)
                for i in range(args["total_sessions"])
            ])
        elif args["dataset"] == "flair":
            self.class_num = args["init_cls"] 
            # initialisation du pool de classifieurs
            self.classifier_pool = nn.ModuleList([DecoderLinear(self.class_num, patch_size = 16, d_encoder=args["embd_dim"]) 
                                                  for i in range(args["total_sessions"])])

        else:
            raise ValueError('Unknown datasets: {}.'.format(args["dataset"]))
        # initialisation des prompts pool
        self.prompt_pool = nn.ModuleList([
            nn.Linear(args["embd_dim"], args["prompt_length"], bias=False)
            for i in range(args["total_sessions"])
        ])
        self.numtask = 0

    @property
    def feature_dim(self):
        return self.image_encoder.out_dim

    def extract_vector(self, image):
        # Ensure the input tensor is on the same device as the model's parameters (e.g., GPU)
        image_encoder = self.image_encoder.to(image.device)
        image_features = image_encoder(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def forward(self, image):
        H_ori, W_ori = image.size(2), image.size(3)
        image = padding(image, patch_size=16)
        H, W = image.size(2), image.size(3)

        logits = []
        masks = []
        image_features = self.image_encoder(image, self.prompt_pool[self.numtask-1].weight)
        for decoder in [self.classifier_pool[self.numtask-1]]:
            logits.append(decoder(image_features, (H,W)))
            mask = F.interpolate(decoder(image_features, (H,W)), size=(H, W), mode="bilinear")
            masks.append(unpadding(mask, (H_ori, W_ori)))
        return {
            'masks' : torch.cat(masks, dim=1),
            'logits': torch.cat(logits, dim=1),
            'features': image_features
        }

    def interface(self, image, selection):
       
        instance_batch = torch.stack([i.weight for i in self.prompt_pool], 0)[selection, :, :]
        image_features = self.image_encoder(image, instance_batch)
        logits = []
        for prompt in self.classifier_pool:
            logits.append(prompt(image_features))

        logits = torch.cat(logits,1)
        selectedlogit = []
        for idx, ii in enumerate(selection):
            selectedlogit.append(logits[idx][self.class_num*ii:self.class_num*ii+self.class_num])
        selectedlogit = torch.stack(selectedlogit)
        return selectedlogit

    def update_fc(self, nb_classes):
        self.numtask +=1

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self
