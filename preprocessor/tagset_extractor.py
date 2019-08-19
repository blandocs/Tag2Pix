import argparse, os, traceback
import json
import pickle
import shutil

from pathlib import Path

SIMPLE_BACKGROUND = 412368
WHITE_BACKGROUND = 515193
sketch_tags = [513837, 1931] # grayscale, sketch
include_tags = [470575, 540830] # 1girl, 1boy
hair_tags = [87788, 16867, 13200, 10953, 16442, 11429, 15425, 8388, 5403, 16581, 87676, 16580, 94007, 403081, 468534]
eye_tags = [10959, 8526, 16578, 10960, 15654, 89189, 16750, 13199, 89368, 95405, 89228, 390186]
blacklist_tags = [63, 4751, 12650, 172609, 555246, 513475] # comic, photo, subtitled, english, black border, multiple_views

def load_metafile_list(path, use_2017=False):
    if use_2017:
        path = path / '2017'
    file_list = [p for p in path.iterdir() if p.is_file()]
    return [f for f in file_list if f.name.startswith('20')]

def make_tag_dict(file_list):
    tag_dict = {}

    def update_tag_dict(meta_obj):
        if "tags" not in meta_obj:
            return
        for tag in meta_obj["tags"]:
            if tag["category"] != "0":
                continue
            tag_id = int(tag["id"])
            if tag_id not in tag_dict:
                tag_dict[tag_id] = [tag["name"], 1]
            else:
                tag_dict[tag_id][1] += 1

    for p in file_list:
        print(f'get jsons in {p.absolute()}')
        try:
            with p.open('r', encoding='utf-8') as f:
                for line in f:
                    meta_obj = json.loads(line)
                    update_tag_dict(meta_obj)
        except Exception:
            print(f'json reader failed: {p.absolute()}')
            traceback.print_exc()
    
    return tag_dict


def main_tag_extract(dataset_path, file_list):
    file_resolutions = {}
    IS_SKETCH = 5

    def json_filter(meta):
        w, h = int(meta['image_width']), int(meta['image_height'])
        if w < 512 and h < 512:
            return False
        if not (3 / 4 < w / h < 4 / 3):
            return False

        tags = set(int(tag['id']) for tag in meta['tags'] if tag['category'] == '0')
        
        for black in blacklist_tags:
            if black in tags:
                return False
        
        if len(tags.intersection(sketch_tags)) >= 1 and WHITE_BACKGROUND in tags:
            if all(len(tags.intersection(lst)) == 0 for lst in [hair_tags, eye_tags]):
                return IS_SKETCH # sketches
            else:
                return False

        if SIMPLE_BACKGROUND not in tags:
            return False
        
        conditions = all(len(tags.intersection(lst)) == 1 for lst in [include_tags, hair_tags, eye_tags])
        if not conditions:
            return False

        return True
    
    tag_lines = []
    img_root_path = dataset_path / '512px'

    save_path = dataset_path / 'train_image_base'
    if not save_path.exists():
        save_path.mkdir()

    sketch_path = dataset_path / 'liner_test'
    if not sketch_path.exists():
        sketch_path.mkdir()

    print('copying images...')
    for p in file_list:
        print(f'get jsons in {p.absolute()}')
        try:
            with p.open('r', encoding='utf-8') as f:
                for line in f:
                    meta_obj = json.loads(line)
                    filtered = json_filter(meta_obj)

                    if filtered is False:
                        continue

                    file_id = int(meta_obj['id'])
                    file_path = img_root_path / f'{file_id%1000:04d}' / f'{file_id}.jpg'

                    if not file_path.exists():
                        continue

                    if filtered == IS_SKETCH:
                        shutil.copy2(file_path, sketch_path)
                    else:
                        shutil.copy2(file_path, save_path)

                    tag_line = ' '.join(tag['id'] for tag in meta_obj['tags'] if tag['category'] == '0')
                    tag_lines.append(f'{file_id} {tag_line}\n')

                    file_resolutions[int(meta_obj['id'])] = (int(meta_obj['image_width']), int(meta_obj['image_height']))
        except Exception:
            print(f'json reader failed: {p.absolute()}')
            traceback.print_exc()
    
    with (dataset_path / 'tags.txt').open('w') as f:
        for tag_line in tag_lines:
            f.write(tag_line)

    with (dataset_path / 'resolutions.pkl').open('wb') as f:
        pickle.dump(file_resolutions, f)

if __name__=='__main__':
    desc = "tag2pix tagset extractor"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--dataset_path', type=str, default='./dataset',
                        help='path to dataset directory')
    parser.add_argument('--use_2017', action='store_true', help='Use Danbooru2017 metadata')
    parser.add_argument('--make_tag_dict', action='store_true',
                        help='make (tag_id - tag_name - count) text to dataset_path/tags_conut.txt')

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    metafile_list = load_metafile_list(dataset_path / 'metadata', args.use_2017)

    if args.make_tag_dict:
        tag_dict = make_tag_dict(metafile_list)
        tag_tuple = [(i, n, c) for i, [n, c] in tag_dict.items()]
        tag_tuple.sort(key=lambda x: x[2], reverse=True)
        with (dataset_path.parent / 'tags_count.txt').open('w') as f:
            for tag_id, tag_name, tag_count in tag_tuple:
                f.write(f'{tag_id} {tag_name} {tag_count}\n')
    else:
        main_tag_extract(dataset_path, metafile_list)
