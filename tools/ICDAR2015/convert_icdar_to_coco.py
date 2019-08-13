import os
import json
import cv2

def readTxt(txt_path):
    objs = []
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip('\ufeff')
        line = line.strip('\n')
        bbox, text = line.split(',')[:8], line.split(',')[-1]

        bbox = [int(coord) for coord in bbox]

        anno = {'category_id': 1, 'bbox': bbox, 'text': text, \
                'isDifficult': 1 if text=='###' else 0}
        objs.append(anno)
    return objs


data_dir = '/home/xiangyuzhu/data/ICDAR2015/train/'

img_lists = os.listdir(os.path.join(data_dir, 'images'))

res = []

for img_name in img_lists:
    txt_name = 'gt_' + img_name.replace('jpg', 'txt')
    objs = readTxt(data_dir + '/annotations/' + txt_name)
    img = cv2.imread(data_dir + '/images/' + img_name)
    height, width, _ = img.shape
    anno = {'img_name': img_name,
            'height': height,
            'width': width,
            'objs': objs}
    res.append(anno)

with open(data_dir + 'annotations.json', 'w') as f:
    json.dump(res, f)