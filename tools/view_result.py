import os
import tqdm
import glob
import shutil

ori_folder = r'C:\Users\peixixio\Downloads\results\170'
new_folder = r'C:\Users\peixixio\Downloads\results\170_new'
try:
    shutil.rmtree(new_folder)
except:
    pass
os.makedirs(new_folder, exist_ok=True)

for file_folder in tqdm.tqdm(glob.glob(ori_folder + '/*')):
    if '.txt' not in file_folder:
        img_file = os.path.join(file_folder, '0.png')
        image_idx = os.path.basename(file_folder)
        if '_gt' in image_idx:
            new_img_file = os.path.join(new_folder, 'gt_' + image_idx + '.png')
        else:
            new_img_file = os.path.join(new_folder, image_idx + '.png')
        shutil.copy(img_file, new_img_file)
    else:
        file_name = os.path.basename(file_folder)
        new_txt_file = os.path.join(new_folder, file_name)
        shutil.copy(file_folder, new_txt_file)
