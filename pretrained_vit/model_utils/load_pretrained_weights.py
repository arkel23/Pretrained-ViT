import errno
import os
import sys
import warnings
import numpy as np
import torch
from torch.hub import get_dir, urlparse, download_url_to_file, _is_legacy_zip_format, _legacy_zip_load

import pretrained_vit
from .configs import ViTConfig


MODEL_NAMES_DIC = {
    'vit_b16': 'B_16',
    'vit_b32': 'B_32',
    'vit_l16': 'L_16',
    'vit_l32': 'L_32'
}


def jax_to_pytorch(k):
    k = k.replace('Transformer/encoder_norm', 'norm')
    k = k.replace('LayerNorm_0', 'norm1')
    k = k.replace('LayerNorm_2', 'norm2')
    k = k.replace('MlpBlock_3/Dense_0', 'pwff.fc1')
    k = k.replace('MlpBlock_3/Dense_1', 'pwff.fc2')
    k = k.replace('MultiHeadDotProductAttention_1/out', 'proj')
    k = k.replace('MultiHeadDotProductAttention_1/query', 'attn.proj_q')
    k = k.replace('MultiHeadDotProductAttention_1/key', 'attn.proj_k')
    k = k.replace('MultiHeadDotProductAttention_1/value', 'attn.proj_v')
    k = k.replace('Transformer/posembed_input', 'positional_embedding')
    k = k.replace('encoderblock_', 'blocks.')
    k = 'patch_embedding.bias' if k == 'embedding/bias' else k
    k = 'patch_embedding.weight' if k == 'embedding/kernel' else k
    k = 'class_token' if k == 'cls' else k
    k = k.replace('head', 'fc')
    k = k.replace('kernel', 'weight')
    k = k.replace('scale', 'weight')
    k = k.replace('/', '.')
    k = k.lower()
    return k


def convert(npz, state_dict):
    new_state_dict = {}
    pytorch_k2v = {jax_to_pytorch(k): v for k, v in npz.items()}
    for pytorch_k, pytorch_v in state_dict.items():

        # Naming
        if 'self_attn.out_proj.weight' in pytorch_k:
            v = pytorch_k2v[pytorch_k]
            v = v.reshape(v.shape[0] * v.shape[1], v.shape[2])
        elif 'self_attn.in_proj_' in pytorch_k:
            v = np.stack((pytorch_k2v[pytorch_k + '*q'],
                          pytorch_k2v[pytorch_k + '*k'],
                          pytorch_k2v[pytorch_k + '*v']), axis=0)
        else:
            if pytorch_k not in pytorch_k2v:
                print(pytorch_k, list(pytorch_k2v.keys()))
                assert False
            v = pytorch_k2v[pytorch_k]
        v = torch.from_numpy(v)

        # Sizing
        if '.weight' in pytorch_k:
            if len(pytorch_v.shape) == 2:
                v = v.transpose(0, 1)
            if len(pytorch_v.shape) == 4:
                v = v.permute(3, 2, 0, 1)
        if ('proj.weight' in pytorch_k):
            v = v.transpose(0, 1)
            v = v.reshape(-1, v.shape[-1]).T
        if ('attn.proj_' in pytorch_k and 'weight' in pytorch_k):
            v = v.permute(0, 2, 1)
            v = v.reshape(-1, v.shape[-1])
        if 'attn.proj_' in pytorch_k and 'bias' in pytorch_k:
            v = v.reshape(-1)
        new_state_dict[pytorch_k] = v
    return new_state_dict


def convert_single(name, filename):
    # Load Jax weights
    npz = np.load(filename)

    # Load PyTorch model
    load_repr_layer = False if 'in1k' in filename else True
    image_size = 384 if 'in1k' in filename else 224
    num_classes = 1000 if 'in1k' in filename else 21843

    cfg = ViTConfig(model_name=name, load_repr_layer=load_repr_layer,
                    image_size=image_size, num_classes=num_classes)
    model = pretrained_vit.ViT(cfg, pretrained=False)

    # Convert weights
    new_state_dict = convert(npz, model.state_dict())

    # Load into model and test
    model.load_state_dict(new_state_dict)

    # Save weights
    new_filename = os.path.abspath(
        '{}.pth'.format(os.path.splitext(filename)[0]))
    torch.save(new_state_dict, new_filename,
               _use_new_zipfile_serialization=False)
    print(f"Converted {filename} and saved to {new_filename}")
    return new_filename


def dl_load_state_dict(model_name, num_classes=21843, image_size=224, url=None, model_dir=None,
                       map_location=None, file_name=None, progress=True):
    # Issue warning to move data if old env is set
    if os.getenv('TORCH_MODEL_ZOO'):
        warnings.warn(
            'TORCH_MODEL_ZOO is deprecated, please use env TORCH_HOME instead')

    if model_dir is None:
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')

    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    if model_name in MODEL_NAMES_DIC:
        model_name_og = model_name
        model_name = MODEL_NAMES_DIC[model_name]
        if num_classes == 1000 and image_size == 384:
            model_name = model_name + '_in1k'
        file_name = model_name
    else:
        raise NotImplementedError

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name

    # cached_file doesnt have extension so it doesnt differentiate between pth and npz files
    cached_file = os.path.join(model_dir, filename)
    cached_file_pth = '{}.pth'.format(cached_file)
    if not os.path.exists(cached_file_pth):
        cached_file_npz = '{}.npz'.format(cached_file)
        sys.stderr.write(
            'Downloading: "{}" to {}\n'.format(url, cached_file_npz))
        download_url_to_file(url, cached_file_npz, progress=progress)
        cached_file = convert_single(name=model_name_og, filename=cached_file_npz)
    if _is_legacy_zip_format(cached_file_pth):
        return _legacy_zip_load(cached_file_pth, model_dir, map_location)

    fp = f'{os.path.join(model_dir, model_name)}.pth'
    if not os.path.exists(fp):
        raise FileNotFoundError

    return torch.load(fp, map_location=map_location)


def load_pretrained_weights(
    model,
    config,
    model_name=None,
    weights_path=None,
    strict=False,
    verbose=True,
):
    """Loads pretrained weights from weights path or download using url.
    Args:
        model (Module): Full model (a nn.Module)
        model_name (str): Model name (e.g. B_16)
        weights_path (None or str):
            str: path to pretrained weights file on the local disk.
            None: use pretrained weights downloaded from the Internet.
        verbose (bool): Whether to print on completion
    """
    load_class_token = hasattr(model, 'class_token')
    load_patch_embedding = (
        hasattr(model, 'patch_embedding') and config.num_channels == 3)
    load_pos_embedding = hasattr(model, 'positional_embedding')
    load_encoder_norm = hasattr(model, 'norm')
    load_fc = hasattr(model, 'fc') and config.num_classes in (21843, 1000)
    load_pre_logits = hasattr(model, 'pre_logits')

    # Load or download weights
    if weights_path is None:
        url = config.url
        state_dict = dl_load_state_dict(model_name, config.num_classes, config.image_size, url)
    else:
        state_dict = torch.load(weights_path)

    has_class_token = True if 'class_token' in state_dict else False

    # Modifications to load partial state dict
    expected_missing_keys = []
    if 'pre_logits.weight' in state_dict.keys() and not load_pre_logits:
        expected_missing_keys += ['pre_logits.weight', 'pre_logits.bias']
    if 'fc.weight' in state_dict.keys() and not load_fc:
        expected_missing_keys += ['fc.weight', 'fc.bias']
    if 'class_token' in state_dict.keys() and not load_class_token:
        expected_missing_keys += ['class_token']
    if 'patch_embedding.weight' in state_dict.keys() and not load_patch_embedding:
        expected_missing_keys += ['patch_embedding.weight',
                                  'patch_embedding.bias']
    if 'positional_embedding.pos_embedding' in state_dict.keys() and not load_pos_embedding:
        expected_missing_keys += ['positional_embedding.pos_embedding']
    if 'norm.weight' in state_dict.keys() and not load_encoder_norm:
        expected_missing_keys += ['norm.weight', 'norm.bias']
    for key in expected_missing_keys:
        state_dict.pop(key)

    # Change size of positional embeddings
    if load_pos_embedding:
        resize_emb = state_dict['positional_embedding.pos_embedding'].shape != \
            model.state_dict()['positional_embedding.pos_embedding'].shape
        if resize_emb:
            posemb = state_dict['positional_embedding.pos_embedding']
            posemb_new = model.state_dict(
            )['positional_embedding.pos_embedding']
            state_dict['positional_embedding.pos_embedding'] = \
                resize_pos_embedding(posemb=posemb, posemb_new=posemb_new,
                                     has_class_token=has_class_token,
                                     load_class_token=load_class_token)
            maybe_print('Resized positional embeddings from {} to {}'.format(
                        posemb.shape, posemb_new.shape), verbose)

    # Load state dict
    ret = model.load_state_dict(state_dict, strict=False)
    if strict:
        for key in ret.missing_keys:
            assert key in expected_missing_keys, '''
            Missing keys when loading pretrained weights: {}
            Expected missing keys: {}
            '''.format(ret.missing_keys, expected_missing_keys)
        assert not ret.unexpected_keys, \
            '''Unexpected keys when loading pretrained weights: {}
            '''.format(ret.unexpected_keys)
        maybe_print('Loaded pretrained weights.', verbose)
    else:
        maybe_print('''Missing keys when loading pretrained weights: {}
            Expected missing keys: {}
            '''.format(ret.missing_keys, expected_missing_keys), verbose)
        maybe_print('''Unexpected keys when loading pretrained weights: {}
            '''.format(ret.unexpected_keys), verbose)
        maybe_print('Loaded pretrained weights.', verbose)
        return ret


def maybe_print(s: str, flag: bool):
    if flag:
        print(s)


def resize_pos_embedding(posemb, posemb_new, has_class_token=True,
                         load_class_token=True):
    """Rescale the grid of position embeddings in a sensible manner"""
    from scipy.ndimage import zoom

    # Deal with class token
    ntok_new = posemb_new.shape[1]
    if load_class_token:
        ntok_new -= 1
    if has_class_token:  # this means classifier == 'token'
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
    else:
        posemb_tok, posemb_grid = posemb[:, :], posemb[1:]

    # Get old and new grid sizes
    gs_old = int(np.sqrt(len(posemb_grid)))
    gs_new = int(np.sqrt(ntok_new))
    posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

    # Rescale grid
    zoom_factor = (gs_new / gs_old, gs_new / gs_old, 1)
    posemb_grid = zoom(posemb_grid, zoom_factor, order=1)
    posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
    posemb_grid = torch.from_numpy(posemb_grid)

    # Deal with class token and return
    if load_class_token:
        posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    else:
        posemb = posemb_grid
    return posemb
