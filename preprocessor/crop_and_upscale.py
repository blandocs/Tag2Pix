import os, math, argparse
import platform, subprocess
import random
import pickle
import shutil
import urllib3
from zipfile import ZipFile
from pathlib import Path

from tqdm import tqdm
import numpy as np
import cv2

WAIFU2X_CAFFE_URL = 'http://github.com/lltcggie/waifu2x-caffe/releases/download/1.2.0.2/waifu2x-caffe.zip'

def is_white(vect):
    return np.all(vect > 250)

def make_square_by_mirror(img, orig_h, orig_w):
    # img shape should be square
    is_bgr = len(img.shape) == 3
    img_h, img_w = img.shape[:2]
    
    if orig_h > orig_w:
        h = img_h
        w = int(orig_w * (img_h / orig_h)) - 2 
    else:
        w = img_w
        h = int(orig_h * (img_w / orig_w)) - 2
    
    crop_h = (img_h - h) // 2
    crop_w = (img_w - w) // 2
    
    img = img[crop_h:crop_h+h, crop_w:crop_w+w]
    diff = max(img_h - img.shape[0], img_w - img.shape[1]) 

    pad_l = diff // 2
    pad_r = diff - pad_l 
    if is_bgr:
        if h > w:
            pad_width = ((0, 0), (pad_l, pad_r), (0, 0))
        else:
            pad_width = ((diff, 0), (0, 0), (0, 0)) # do not reflect bottom part of torso 
    else:
        if h > w:
            pad_width = ((0, 0), (pad_l, pad_r))
        else:
            pad_width = ((diff, 0), (0, 0))

    if h != w:
        return np.pad(img, pad_width, mode='symmetric')
    else:
        return img


def delete_img(img_path):
    if img_path.exists():
        img_path.unlink()

def crop_all(dataset_path):
    train_base = dataset_path / 'train_image_base'
    liner_base = dataset_path / 'liner_test'
    save_base = dataset_path / 'rgb_cropped'

    if not save_base.exists():
        save_base.mkdir()

    with (dataset_path / 'resolutions.pkl').open('rb') as f:
        resolutions = pickle.load(f)

    print('cropping train_image_base...')
    img_len = len(list(train_base.iterdir()))
    for img_f in tqdm(train_base.iterdir(), total=img_len):
        file_id = int(img_f.stem)
        if file_id not in resolutions:
            print(f'{file_id} not found in resolutions.pkl')
            continue

        w, h = resolutions[file_id]

        img = cv2.imread(str(img_f), cv2.IMREAD_COLOR)
        cropped_img = make_square_by_mirror(img, h, w)
        cv2.imwrite(str(save_base / (img_f.stem + '.png')), cropped_img)
        
    print('cropping liner_test...')
    liner_list = list(liner_base.iterdir())
    for img_f in tqdm(liner_list):
        file_id = int(img_f.stem)
        if file_id not in resolutions:
            print(f'{file_id} not found in resolutions.pkl')
            h = 512, 512

        w, h = resolutions[file_id]

        img = cv2.imread(str(img_f), cv2.IMREAD_COLOR)
        cropped_img = make_square_by_mirror(img, h, w)
        delete_img(img_f)
        cv2.imwrite(str(liner_base / (img_f.stem + '.png')), cropped_img)
        

    img_list = list(save_base.iterdir())
    random.shuffle(img_list)
    benchmark_dir = dataset_path / 'benchmark'
    if not benchmark_dir.exists():
        benchmark_dir.mkdir()
    for img_f in img_list[:32]:
        shutil.move(str(img_f.absolute()), str((benchmark_dir / img_f.name).absolute()))

def upscale_waifu2x_all(dataset_path, image_base, save_path):
    curr_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    waifu_path = curr_dir / 'waifu2x-caffe'
      

    print('upscaling with waifu2x...')
    subprocess.run([str(waifu_path / 'waifu2x-caffe-cui.exe'), '-i', str(image_base.absolute()), '-w', 
            '768', '-b', '8', '-n', '1', '-p', 'cudnn', '-o', str(save_path.absolute())])

def upscale_lanczos_all(dataset_path, image_base, save_path):
    print('upscaling with lanczos...')
    img_len = len(list(image_base.iterdir()))
    for img_f in tqdm(image_base.iterdir(), total=img_len):
        img = cv2.imread(str(img_f), cv2.IMREAD_COLOR)
        img_up = cv2.resize(img, (768, 768), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(str(save_path / img_f.name), img_up)
    
def upscale_all(dataset_path, image_base=None, save_path=None):
    if image_base is None:
        image_base = dataset_path / 'rgb_cropped'
    if save_path is None:
        save_path = dataset_path / 'rgb_train'

    if not save_path.exists():
        save_path.mkdir()  

    if platform.system() == 'Windows':
        upscale_waifu2x_all(dataset_path, image_base, save_path)
    else:
        upscale_lanczos_all(dataset_path, image_base, save_path)    
    
def download_waifu2x_caffe():
    if platform.system() != 'Windows':
        print("Platform is not Windows. Using LANCZOS filter to upscale images.")
        return
        
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(curr_dir, 'waifu2x-caffe.zip')
    
    if (Path(curr_dir) / 'waifu2x-caffe').exists():
        print('found waifu2x-caffe')
        return

    print('Downloading waifu2x-caffe...')
    http = urllib3.PoolManager()

    with http.request('GET', WAIFU2X_CAFFE_URL, preload_content=False) as r, open(save_path, 'wb') as out_file:       
        shutil.copyfileobj(r, out_file)

    print('Unzipping waifu2x-caffe.zip...')

    with ZipFile(save_path, 'r') as zip_obj:
        zip_obj.extractall(curr_dir)

    print('Finished unzipping waifu2x-caffe')
    os.remove(save_path)

if __name__=='__main__':
    desc = "tag2pix tagset extractor"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--dataset_path', type=str, default='./dataset',
                        help='path to dataset directory')
    parser.add_argument('--download_waifu2x_caffe', action='store_true')
    parser.add_argument('--crop_only', action='store_true',
                        help='only makes cropped image')
    parser.add_argument('--upscale_only', action='store_true',
                        help='only makes upscaled image')

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)

    if args.download_waifu2x_caffe:
        download_waifu2x_caffe()
    else:
        if not args.upscale_only:
            crop_all(dataset_path)
        if not args.crop_only:
            download_waifu2x_caffe()
            upscale_all(dataset_path)
            
