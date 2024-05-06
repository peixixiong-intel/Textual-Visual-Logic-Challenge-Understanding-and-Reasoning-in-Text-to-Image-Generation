import os, glob
import json
import re

split = 'CLEVR'
# file_name = r'C:\Users\peixixio\Downloads\coco\annotation/captions_train2014.json'
file_name = '/net/csr-dgx1-04/data2/peixixio/coco/annotation/captions_test2014.json'
# file_name = '/net/csr-dgx1-04/data2/peixixio/coco/annotation/captions_train2014.json'

# split = 'CoDraw'
# file_name = r'C:\Users\peixixio\Downloads\coco_codraw\annotation/captions_train2014.json'

clevr_class = '/net/csr-dgx1-04/data2/peixixio/GeNeVA_datasets/data/objects.txt'
codraw_class = r'C:\Users\peixixio\Downloads\coco_codraw\objects.txt'

if split == 'CLEVR':
    obj_class = clevr_class
else:
    obj_class = codraw_class

with open(obj_class) as f:
    obj_category_ = [line.rstrip() for line in f]

obj_category = []
if split == 'CLEVR':
    for obj in obj_category_:
        entity, attr = obj.split(' ')
        new_name = attr + ' ' + entity
        obj_category.append(new_name)
else:
    for obj in obj_category_:
        obj_category.append(obj)
print(obj_category)

if split == 'CLEVR':
    rel2numb = {'left': 0, 'right': 0, 'front': 0, 'behind': 0}
else:
    rel_print = {'directional': ['left', 'right', 'front', 'behind', 'under', 'above', 'after', 'on'],
                 'distance': ['further', 'closer'],
                 'positional': ['upper', 'lower', 'higher'],
                 'attributes comparision': ['taller', 'shorter', 'darker', 'brighter', 'bigger', 'smaller', 'larger'],
                 'actions': [
    'slide', 'handlers', 'stands', 'cut', 'got', 'faces', 'wears', 'move',
    'go', 'facing', 'covered', 'drink', 'show', 'reread', 'are', 'turn', 'has',
    'be', 'raining', 'covers', 'looking', 'want', 'look', 'add', 'stay', 'start',
    'starting', 'touching', 'sitting', 'showing', 'adjust', 'check', 'sit', 'make',
    'slides', 'makes',
]}
    rel2numb = {}
    for rel_cat in rel_print:
        for rel in rel_print[rel_cat]:
            rel2numb[rel] = 0
    rel_table = {key:0 for key in rel_print}


obj2numb = {obj_name:0 for obj_name in obj_category}

for annotation in json.load(open(file_name))['annotations']:
    caption = annotation['caption']
    for obj in obj2numb.keys():
        if obj in caption:
            obj2numb[obj] += 1
    for rel in rel2numb.keys():
        if rel in caption:
            numb = len(re.findall(rel, caption))
            rel2numb[rel] += numb

if split == 'CLEVR':
    print(obj2numb)
    print(rel2numb)
    print(len(rel2numb))
else:
    new_print = [(value, key) for key, value in zip(obj2numb.keys(), obj2numb.values())]
    new_count = sorted(new_print, key=lambda x: int(x[0]), reverse=True)
    new_count = [(name, cnt) for cnt, name in new_count][:35]
    print_cnt = {name: cnt for name, cnt in new_count}
    print (print_cnt)

    for rel_cat in rel_print:
        cnt = 0
        for rel in rel_print[rel_cat]:
            cnt += rel2numb[rel]
        rel_table[rel_cat] = cnt
    print(rel_table)

