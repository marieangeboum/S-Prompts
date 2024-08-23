import torch
import torch.nn as nn
import copy
import timm

from models.vit import VisionTransformer, PatchEmbed,resolve_pretrained_cfg, build_model_with_cfg, checkpoint_filter_fn

from models.mae_vit import ViT_Win_RVSA

def _create_vision_transformer(nb_class, variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    # NOTE this extra code to support handling of repr size for in21k pretrained models
    pretrained_cfg = resolve_pretrained_cfg(variant)
    print(pretrained_cfg)
    # default_num_classes = pretrained_cfg['num_classes']
    # default_num_classes = pretrained_cfg.num_classes
    # num_classes = kwargs.get('num_classes', default_num_classes)
    # repr_size = kwargs.pop('representation_size', None)
    # if repr_size is not None and num_classes != default_num_classes:
    #     repr_size = None
    print(pretrained_cfg['url'])
    model = ViT_Win_RVSA(num_classes=nb_class)
    model.load_state_dict(torch.load('/d/maboum/S-Prompts/models/vit_rvsa_ucm55.pth')["model"],strict=False)    
    # model = build_model_with_cfg(
    #     MaskedAutoencoderViT, variant, pretrained,
    #     pretrained_cfg=pretrained_cfg,
    #     representation_size=repr_size,
        
    #     pretrained_filter_fn=checkpoint_filter_fn,
    #     pretrained_custom_load='npz' in pretrained_cfg['url'],
    #     **kwargs)
    
    return model

class MAEViTNet(nn.Module):

    def __init__(self, args):
        super(MAEViTNet, self).__init__()
        self._device = args['device'][0]
        model_kwargs = dict(patch_size=16, embed_dim=args["embd_dim"], depth=12, num_heads=12)
#        self.image_encoder =_create_vision_transformer('vit_base_patch16_224_remote_sensing', pretrained=True, **model_kwargs)
        self.image_encoder =_create_vision_transformer(args["init_cls"],args["variant"], pretrained=True, **model_kwargs)
        # a changer pour avoir
        self.class_num = args["init_cls"] 
        # initialisation du pool de classifieurs
        # self.classifier_pool = nn.Linear(args["embd_dim"], self.class_num, bias=True)
        self.classifier_pool = nn.Identity()
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
        logits = []
        # image_features = self.image_encoder(image, self.prompt_pool[self.numtask-1].weight)
        image_features = self.image_encoder(image)
        prompts = self.classifier_pool
        logits=prompts(image_features)

        return logits

    def interface(self, image, selection):
       
        # instance_batch = torch.stack([i.weight for i in self.prompt_pool], 0)[selection, :, :]
        # image_features = self.image_encoder(image, instance_batch)
        image_features = self.image_encoder(image)
        logits = []
        
        prompts = self.classifier_pool
        logits=prompts(image_features)
        
        # selectedlogit = []
        # for idx, ii in enumerate(selection):
        #     selectedlogit.append(logits[idx][self.class_num*ii:self.class_num*ii+self.class_num])
        # selectedlogit = torch.stack(selectedlogit)
        return logits

    def update_fc(self, nb_classes):
        self.numtask +=1

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self
