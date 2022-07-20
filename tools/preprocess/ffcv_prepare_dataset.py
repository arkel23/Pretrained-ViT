import os

from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField

from pretrained_vit.data_utils.datasets import get_set
from pretrained_vit.other_utils.build_args import parse_ffcv_prepare_dataset_args


def convert_datasets(args):

    if args.dataset_name in ['imagenet']:
        write_mode = 'smart'
    else:
        write_mode = 'raw'

    for split, fn in zip(('train', 'val', 'test'), (args.df_train, args.df_val, args.df_test)):
        print(f'Started converting split: {split} file: {fn}')
        dataset = get_set(args, split)
        dataset.transform = None
        save_path = os.path.join(args.dataset_root_path, fn.replace('.csv', '.beton'))

        writer = DatasetWriter(
            save_path, {
                # Tune options to optimize dataset size, throughput at train-time
                'image': RGBImageField(
                    write_mode=write_mode,
                    max_resolution=args.max_resolution,
                    compress_probability=0.5,
                    jpeg_quality=90),
                'label': IntField()
            })
        writer.from_indexed_dataset(dataset)
        print(f'Finished converting: {args.dataset_name} {split} split and saved to {save_path}')
    return 0


def main():
    args = parse_ffcv_prepare_dataset_args()
    convert_datasets(args)


if __name__ == '__main__':
    main()
