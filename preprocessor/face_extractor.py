import os, argparse, shutil
from pathlib import Path
from tqdm import tqdm

import cv2

CASCADE_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lbpcascade_animeface.xml')


include_tags = [470575, 540830] # 1girl, 1boy
hair_tags = [13197, 15080, 2373, 16751, 380350, 1709, 417660, 6126, 464561, 2355, 3522, 
             2376, 4334, 13804, 15522, 2646, 72, 2785, 400123, 381629, 52138, 652604, 470, 
             442865, 389813, 582201, 464588, 600250, 2279, 10422, 463173, 381555, 2335, 468554, 
             421663, 385639, 457597, 551772, 8091, 479563, 258190, 452195, 464713, 565513, 458482, 
             448225, 524399, 653206, 3649, 9114, 539837, 454933, 448202, 4188, 413908, 2849, 1288118, 
             390681, 445388, 401836, 529256, 534835, 689532, 383851, 1337464, 146061, 389814, 394305, 
             379945, 368, 506510, 87788, 16867, 13200, 10953, 16442, 11429, 15425, 8388, 5403, 16581, 87676, 16580, 94007, 403081, 468534]
eye_tags = [10959, 8526, 16578, 10960, 15654, 89189, 16750, 13199, 89368, 95405, 89228, 390186]
facial_tags = [469576, 1815, 11906, 1304828, 465619, 384553, 658573, 572080, 461042, 6532, 1793, 664375, 
               14599, 6054, 6010, 384884, 10863, 399541, 1445905, 419938, 499624, 5565, 502710, 402217, 374938, 
               575982, 1797, 13227, 462808, 466990, 574407, 2270, 446950, 10231, 376528, 537200, 481508, 4536, 
               385882, 521420, 3988, 16700, 464903, 492380, 1441885, 688777, 404507, 1441886, 473529, 448882, 423620, 
               401137, 572731, 3389, 1373022, 1373029]

def get_filtered_tags(tag_path):
    filtered_tag_dict = dict()
    valid_tag_set = set(include_tags + hair_tags + eye_tags + facial_tags)

    with tag_path.open('r') as f:
        for line in f:
            if line.strip() == '':
                continue
            tags = [int(tag) for tag in line.strip().split(' ')]
            
            file_id, tags = tags[0], tags[1:]
            valid_tags = valid_tag_set.intersection(tags)

            filtered_tag_dict[file_id] = ' '.join(map(str, valid_tags))

    return filtered_tag_dict
            
def detect_all(dataset_path):
    print('detecting/cropping face images')

    cascade = cv2.CascadeClassifier(CASCADE_FILE_PATH)
    train_path = dataset_path / 'rgb_train'
    save_path = dataset_path / 'temp_faces'
    
    if not save_path.exists():
        save_path.mkdir()

    def detect(image_path):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = cascade.detectMultiScale(gray, scaleFactor = 1.2,
                                        minNeighbors = 5, minSize = (48, 48))
                                        
        if len(faces) != 1:
            return None, None
        
        face_pos = faces[0]
        x, y, w, h = face_pos

        w = w if w > h else h
        
        x = max(0, int(x - w * 0.1))
        y = max(0, int(y - w * 0.1))
        w = int(w * 1.2)
        if w < 128:
            return None, None
        
        cropped_img = image[y:y+w, x:x+w]

        oh, ow = image.shape[:2]
        return cropped_img, [x/ow, y/oh, w/oh, w/ow]

    count = 0
    with (save_path / 'face_position.txt').open('w') as f:
        img_list = list(filter(lambda x: int(x.stem) > 0, train_path.iterdir()))
        for img_f in tqdm(img_list):
            if img_f.suffix.lower() not in ['.png', '.jpg', 'jpeg']:
                continue
            
            cropped_img, face_pos = detect(img_f)
            if face_pos is None:
                continue

            cv2.imwrite(str(save_path / ('-' + img_f.name)), cropped_img)
            f.write(f'{img_f.stem} {" ".join(map(str, face_pos))}\n')
            count += 1

    print(f'saved {count} faces')

def get_face_pos(temp_faces_path):
    face_pos_dict = dict()
    with (temp_faces_path / 'face_position.txt').open('r') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            pos = [float(l) for l in line.split(' ')]
            face_pos_dict[int(pos[0])] = pos[1:]

    return face_pos_dict

def move_all(dataset_path):
    print('moving face images')

    temp_faces_path = dataset_path / 'temp_faces'
    train_dirs = [dataset_path / dir_name for dir_name in ['keras_train', 'xdog_train', 'simpl_train']]
    
    tag_path = dataset_path / 'tags.txt'
    face_pos_dict = get_face_pos(temp_faces_path)
    tag_dict = get_filtered_tags(tag_path)
    moved_list = set()

    for train_dir in train_dirs:
        if not train_dir.exists():
            continue
        
        for file_id, [x, y, h, w] in tqdm(face_pos_dict.items(), total=len(face_pos_dict)):
            if file_id not in tag_dict:
                continue
            file_pos = train_dir / f'{file_id}.png'
            if not file_pos.exists():
                continue
            save_pos = train_dir / f'-{file_id}.png'
            img = cv2.imread(str(file_pos), cv2.IMREAD_COLOR)
            oh, ow = img.shape[:2]
            x, y, h, w = round(x*ow), round(y*oh), round(h*oh), round(w*ow)

            cv2.imwrite(str(save_pos), img[y:y+w, x:x+w])
            moved_list.add(file_id)
    
    rgb_path = dataset_path / 'rgb_train'
    for file_id in moved_list:
        file_path = temp_faces_path / f'-{file_id}.png'
        if file_path.exists():
            shutil.move(str(file_path), str(rgb_path / f'-{file_id}.png'))

    with tag_path.open('a') as f:
        for file_id in moved_list:
            tags = tag_dict[file_id]
            f.write(f'-{file_id} {tags}\n')


if __name__ == '__main__':
    desc = "tag2pix face extractor"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--dataset_path', type=str, default='./dataset',
                        help='path to dataset directory')

    parser.add_argument('--add_all', action='store_true',
                        help='Move every face image in the `temp_faces` to `rgb_train`.' +
                             'It also crops face images from sketches and add them to train set')

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)

    if args.add_all:
        move_all(dataset_path)
    else:
        detect_all(dataset_path)