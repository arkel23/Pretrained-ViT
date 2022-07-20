from pretrained_vit import ViT, ViTConfig

models_list = ['vit_b16', 'vit_b32', 'vit_l16', 'vit_l32']
for model_name in models_list:
    cfg = ViTConfig(model_name=model_name, load_repr_layer=True)
    model = ViT(cfg, pretrained=True)
    print(cfg)
    # print(model)
