import os, argparse
import urllib3
import shutil
import random
from multiprocessing import Pool
from pathlib import Path
from itertools import cycle

import cv2
from tqdm import tqdm

from utils.xdog_blend import get_xdog_image, add_intensity


SKETCHKERAS_URL = 'http://github.com/lllyasviel/sketchKeras/releases/download/0.1/mod.h5'

def make_xdog(img):
    s = 0.35 + 0.1 * random.random()
    k = 2 + random.random()
    g = 0.95
    return get_xdog_image(img, sigma=s, k=k, gamma=g, epsilon=-0.5, phi=10**9)

def download_sketchKeras():
    curr_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    save_path = curr_dir / 'utils' / 'sketchKeras.h5'
    
    if save_path.exists():
        print('found sketchKeras.h5')
        return

    print('Downloading sketchKeras...')
    http = urllib3.PoolManager()

    with http.request('GET', SKETCHKERAS_URL, preload_content=False) as r, save_path.open('wb') as out_file:       
        shutil.copyfileobj(r, out_file)

    print('Finished downloading sketchKeras.h5')

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def xdog_write(path_img):
    path, img, xdog_result_path = path_img
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    xdog_img = make_xdog(img)
    cv2.imwrite(str(xdog_result_path / path), xdog_img)

if __name__=='__main__':
    desc = "tag2pix sketch extractor"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--dataset_path', type=str, default='./dataset',
                        help='path to dataset directory')
    parser.add_argument('--xdog_only', action='store_true')
    parser.add_argument('--keras_only', action='store_true')
    parser.add_argument('--no_upscale', action='store_true', help='do not upscale keras_train')

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    keras_result_path = dataset_path / 'keras_train'
    xdog_result_path = dataset_path / 'xdog_train'

    if not keras_result_path.exists():
        keras_result_path.mkdir()
    if not xdog_result_path.exists():
        xdog_result_path.mkdir()

    print('reading images...')
    img_list = []
    path_list = []
    for img_f in (dataset_path / 'rgb_train').iterdir():
        if not img_f.is_file():
            continue
        if img_f.suffix.lower() != '.png':
            continue

        path_list.append(img_f.name)
        img_list.append(str(img_f))

    if not args.xdog_only:
        from utils.sketch_keras_util import batch_keras_enhanced

        download_sketchKeras()

        print('Extracting sketchKeras')
        for p_list, chunk in tqdm(list(zip(chunks(path_list, 16), chunks(img_list, 16)))):
            chunk = list(map(cv2.imread, chunk))
            krs = batch_keras_enhanced(chunk)

            for name, sketch in zip(p_list, krs):
                sketch = add_intensity(sketch, 1.4)
                cv2.imwrite(str(keras_result_path / name), sketch)

        if not args.no_upscale:
            from crop_and_upscale import upscale_all
            print('upscaling keras_train images...')
            moved_temp_keras = dataset_path / 'temp_keras'
            shutil.move(str(keras_result_path), str(moved_temp_keras))
            upscale_all(dataset_path, image_base=moved_temp_keras, save_path=keras_result_path)
            shutil.rmtree(str(moved_temp_keras))
            
        print('extracting sketches from benchmark images...')
        keras_test_dir = dataset_path / 'keras_test'
        if not keras_test_dir.exists():
            keras_test_dir.mkdir()

        benchmark_dir = dataset_path / 'benchmark'
        bench_imgs = list(benchmark_dir.iterdir())
        for img_fs in tqdm(list(chunks(bench_imgs, 16))):
            chunk = list(map(lambda x: cv2.imread(str(x)), img_fs))
            krs = batch_keras_enhanced(chunk)

            for img_f, sketch in zip(img_fs, krs):
                sketch = add_intensity(sketch, 1.4)
                cv2.imwrite(str(keras_test_dir / img_f.name), sketch)

    if not args.keras_only:
        print('Extracting XDoG with 8 threads')
            
        with Pool(8) as p:
            p.map(xdog_write, zip(path_list, img_list, cycle([xdog_result_path])))
    
        
