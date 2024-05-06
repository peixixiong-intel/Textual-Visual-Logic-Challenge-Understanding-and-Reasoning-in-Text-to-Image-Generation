import json
import numpy as np


ann_coco = r'C:\Users\peixixio\Downloads\annotations_trainval2017\annotations/captions_train2017.json'
ann_clevr = r'C:\Users\peixixio\Downloads\coco\annotation/captions_train2014.json'
ann_codraw = r'C:\Users\peixixio\Downloads\coco_codraw\annotation/captions_train2014.json'
ann_VISOR = r'C:\Users\peixixio\Downloads\text_spatial_rel_phrases.json'

length_dict ={'coco': [], 'clevr': [], 'codraw': [], 'VISOR':[]}
# for ann_file, dict_name in zip([ann_coco, ann_clevr, ann_codraw], ['coco', 'clevr', 'codraw']):
# # for ann_file, dict_name in zip([ann_VISOR], ['VISOR']):
#     length_list = []
#     for ann in json.load(open(ann_file))['annotations']:
#         length = len(ann['caption'].split(' '))
#         length_list.append(length)
#     print(ann_file, len(json.load(open(ann_file))['annotations']))
#     length_list = np.array(length_list)
#     length_dict[dict_name] = {'mean': length_list.mean(), 'std': length_list.std(), 'max': length_list.max(), 'min': length_list.min()}
# print(length_dict)

for ann_file, dict_name in zip([ann_VISOR], ['VISOR']):
    length_list = []
    for ann in json.load(open(ann_file)):
        length = len(ann['text'].split(' '))
        length_list.append(length)
        # if length == 2:
        #     print(ann)
    # print(ann_file, len(json.load(open(ann_file))))
    length_list = np.array(length_list)
    length_dict[dict_name] = {'mean': length_list.mean(), 'std': length_list.std(), 'max': length_list.max(), 'min': length_list.min()}

print(length_dict)

'''
{'coco': {'mean': 10.613364021813155, 'std': 2.4368455927168085, 'max': 179, 'min': 8}, 
'clevr': {'mean': 79.95385640513076, 'std': 2.764515948556267, 'max': 88, 'min': 70}, 
'codraw': {'mean': 112.31843785204656, 'std': 46.77797540365004, 'max': 422, 'min': 15}}


'''