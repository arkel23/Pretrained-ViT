# ViT

Forked and modified from [Luke Melas-Kyriazi repository](https://github.com/lukemelas/PyTorch-Pretrained-ViT) to support fully featured and versatile training.

## Setup

```
pip install -e .
python download_convert_models.py # can modify to download different models, by default it downloads all 4 ViTs pretrained on ImageNet21k
```

## Usage
```
import torch
from pretrained_vit import ViT, ViTConfig

model_name = 'vit_b16'
cfg = ViTConfig(model_name, num_classes=1000, classifier='pool')
model = ViT(cfg)

x = torch.rand(2, cfg.num_channels, cfg.image_size, cfg.image_size)
out = model(x)
```

## About

This repository contains an op-for-op PyTorch reimplementation of the [Vision Transformer](https://openreview.net/forum?id=YicbFdNTTy) architecture from [Google](https://github.com/google-research/vision_transformer), along with pre-trained models and examples.


Visual Transformers (ViT) are a straightforward application of the [transformer architecture](https://arxiv.org/abs/1706.03762) to image classification. Even in computer vision, it seems, attention is all you need. 

The ViT architecture works as follows: (1) it considers an image as a 1-dimensional sequence of patches, (2) it prepends a classification token to the sequence, (3) it passes these patches through a transformer encoder (like [BERT](https://arxiv.org/abs/1810.04805)), (4) it passes the first token of the output of the transformer through a small MLP to obtain the classification logits. 
ViT is trained on a large-scale dataset (ImageNet-21k) with a huge amount of compute. 

<div style="text-align: center; padding: 10px">
    <img src="https://raw.githubusercontent.com/google-research/vision_transformer/master/figure1.png" width="100%" style="max-width: 300px; margin: auto"/>
</div>

## Credit

Other great repositories with this model include: 
 - [Google Research's repo](https://github.com/google-research/vision_transformer)
 - [Ross Wightman's repo](https://github.com/rwightman/pytorch-image-models)
 - [Phil Wang's repo](https://github.com/lucidrains/vit-pytorch)
 - [Eunkwang Jeon's repo](https://github.com/jeonsworld/ViT-pytorch)
 - [Luke Melas-Kyriazi repo](https://github.com/lukemelas/PyTorch-Pretrained-ViT)

## Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.

I look forward to seeing what the community does with these models!


