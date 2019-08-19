import argparse
import os
import pprint
from pathlib import Path

import torch

from tag2pix import tag2pix

DATASET_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset')
TAG_FILE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset', 'tag_dump.pkl')
PRETRAIN_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model.pth')

def parse_args():
    desc = "tag2pix: Line Art Colorization using Text Tag"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--model', type=str, default='tag2pix', choices=['tag2pix', 'senet', 'resnext', 'catconv', 'catall', 'adain', 'seadain'],
                        help='Model Types. (default: tag2pix == SECat)')

    parser.add_argument('--cpu', action='store_true', help='If set, use cpu only')
    parser.add_argument('--test', action='store_true', help='Colorize line arts in test_dir based on `tag_txt`')
    parser.add_argument('--save_freq', type=int, default=10, 
        help='Save network dump by every `save_freq` epoch. if set to 0, save the last result only')
    
    parser.add_argument('--thread', type=int, default=8, help='total thread count of data loader')
    parser.add_argument('--epoch', type=int, default=20, help='The number of epochs to run')

    parser.add_argument('--batch_size', type=int, default=64, help='Total batch size')
    parser.add_argument('--input_size', type=int, default=256, help='Width / Height of input image (must be rectangular)')
    parser.add_argument('--data_size', default=0, type=int, help='Total training image count. if 0, use every train data')
    parser.add_argument('--test_image_count', type=int, default=64, help='Total count of colorizing test images')

    parser.add_argument('--data_dir', default=DATASET_PATH, help='Path to the train/test data root directory')
    parser.add_argument('--test_dir', type=str, default='liner_test', help='Directory name of the test line art directory. It has to be in the data_dir.')
    parser.add_argument('--tag_txt', type=str, default='tags.txt', help='Text file name of formatted text tag sets (see README). It has to be in the data_dir.')

    parser.add_argument('--result_dir', type=str, default='./results', help='Path to save generated images and network dump')
    parser.add_argument('--pretrain_dump', default=PRETRAIN_PATH, help='Path of pretrained CIT network dump.')
    parser.add_argument('--tag_dump', default=TAG_FILE_PATH, help='Path of tag dictionary / metadata pickle file.')
    parser.add_argument('--load', type=str, default="", help='Path to load network weights (if non-empty)')

    parser.add_argument('--lrG', type=float, default=0.0002, help='Learning rate of tag2pix generator')
    parser.add_argument('--lrD', type=float, default=0.0002, help='Learning rate of tag2pix discriminator')
    parser.add_argument('--l1_lambda', type=float, default=1000, help='Coefficient of content loss')
    parser.add_argument('--guide_beta', type=float, default=0.9, help='Coefficient of guide decoder')
    parser.add_argument('--adv_lambda', type=float, default=1, help='Coefficient of adversarial loss')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam optimizer parameter')
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam optimizer parameter')
    parser.add_argument('--color_space', type=str, default='rgb', choices=['lab', 'rgb', 'hsv'], help='color space of images')
    parser.add_argument('--layers', type=int, nargs='+', default=[12,8,5,5],
        help='Block counts of each U-Net Decoder blocks of generator. The first argument is count of bottom block.')

    parser.add_argument('--cit_cvt_weight', type=float, nargs='+', default=[1, 1], help='CIT CVT Loss weight. space-separated')
    parser.add_argument('--two_step_epoch', type=int, default=0,
        help='If nonzero, apply two-step train. (start_epoch to args.auto_two_step_epoch: cit_cvt_weight==[0, 0], after: --cit_cvt_weight)')
    parser.add_argument('--brightness_epoch', type=int, default=0,
        help='If nonzero, control brightness after this epoch (see Section 4.3.3)' +
             '(start_epoch to bright_down_epoch: ColorJitter.brightness == 0.2, after: [1, 7])')
    parser.add_argument('--save_all_epoch', type=int, default=0,
        help='If nonzero, save network dump by every epoch after this epoch')

    parser.add_argument('--use_relu', action='store_true', help='Apply ReLU to colorFC')
    parser.add_argument('--no_bn', action='store_true', help='Remove every BN Layer from Generator')
    parser.add_argument('--no_guide', action='store_true', help='Remove guide decoder from Generator. If set, Generator will return same G_f: like (G_f, G_f)')
    parser.add_argument('--no_cit', action='store_true', help='Remove pretrain CIT Network from Generator')


    parser.add_argument('--seed', type=int, default=-1, help='if positive, apply random seed')

    args = parser.parse_args()
    validate_args(args)

    return args

def validate_args(args):
    print('validating arguments...')
    pprint.pprint(args.__dict__)

    assert args.epoch >= 1, 'number of epochs must be larger than or equal to one'
    assert args.batch_size >= 1, 'batch size must be larger than or equal to one'
    
    if args.load != '':
        assert os.path.exists(args.load), 'cannot find network dump file'
    assert os.path.exists(args.pretrain_dump), 'cannot find pretrained CIT network dump file'
    assert os.path.exists(args.tag_dump), 'cannot find tag metadata pickle file'

    data_dir_path = Path(args.data_dir)

    assert data_dir_path.exists(), 'cannot find train/test root directory'
    assert (data_dir_path / args.tag_txt).exists(), 'cannot find formatted text tag file'
    # if args.test_image_count > 0:
    #     assert (data_dir_path / args.test_dir).exists(), 'cannot find test directory'

    result_dir_path = Path(args.result_dir)
    if not result_dir_path.exists():
        result_dir_path.mkdir()

def main():
    args = parse_args()
    if not args.cpu and args.seed < 0:
        torch.backends.cudnn.benchmark = True

    gan = tag2pix(args)

    if args.test:
        gan.test()
        print(" [*] Testing finished!")
    else:
        gan.train()
        gan.visualize_results(args.epoch)
        print(" [*] Training finished!")

if __name__ == '__main__':
    main()
