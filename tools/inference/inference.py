import os
import glob
import json
import argparse

import torch
from torchvision import transforms
from PIL import Image

from pretrained_vit import ViT, ViTConfig


def prepare_img(fn, image_size):
    # open img
    img = Image.open(fn).convert('RGB')
    # Preprocess image
    tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = tfms(img).unsqueeze(0)
    return img


def search_images(args):
    # if path is a file
    if os.path.isfile(args.images_path):
        return [args.images_path]
    # else if directory
    # the tuple of file types
    types = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    files_all = []
    for file_type in types:
        # files_all is the list of files
        path = os.path.join(args.images_path, '**', file_type)
        files_curr_type = glob.glob(path, recursive=True)
        files_all.extend(files_curr_type)

        print(file_type, len(files_curr_type))

    print('Total image files pre-filtering', len(files_all))
    return files_all


def inference(args):
    files_all = search_images(args)

    cfg = ViTConfig(args.model_name, image_size=384, num_classes=1000)
    model = ViT(cfg, pretrained=True)
    model.eval()

    # Load class names
    labels_map = json.load(open('labels_map.txt'))
    labels_map = [labels_map[str(i)] for i in range(cfg.num_classes)]

    for file in files_all:
        print(file)
        img = prepare_img(file, cfg.image_size)

        # Classify
        with torch.no_grad():
            outputs = model(img).squeeze(0)

        for idx in torch.topk(outputs, k=3).indices.tolist():
            prob = torch.softmax(outputs, -1)[idx].item()
            print('[{idx}] {label:<75} ({p:.2f}%)'.format(idx=idx, label=labels_map[idx], p=prob*100))

    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path', type=str, default='samples',
                        help='path to folder (with images) or image')
    parser.add_argument('--model_name', type=str, default='vit_b16')
    args = parser.parse_args()

    inference(args)
    return 0


if __name__ == '__main__':
    main()
