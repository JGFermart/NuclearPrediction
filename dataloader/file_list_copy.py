import argparse
import os
import glob
import sys

parser = argparse.ArgumentParser(description='datasets list')
# modify this to point to the folder
parser.add_argument('--img_path', type=str, default='/media/jyo/9CAE3358AE332A62/Project_NTU/3T3')
parser.add_argument('--out_txt_prefix', default='/home/jyo/code/SUTD_Cell/datasets/cell_3T3_sampled')
args = parser.parse_args()

img_name = '/*/*.tif'
img_files = sorted(glob.glob(os.path.join(args.img_path+img_name)))
num_img = len(img_files)

txt_img_A = args.out_txt_prefix+'_test_A.txt'
txt_img_M = args.out_txt_prefix+'_test_M.txt'
txt_img_N = args.out_txt_prefix+'_test_N.txt'
txt_files_A = []
txt_files_M = []
txt_files_N = []

for img_idx, img_fn in enumerate(img_files):
    print("\rProgress:{0:<5}% (image:{1})".format(round(100. * (img_idx + 1.) / num_img, 2), img_fn),
          sys.stdout.flush())
    # manually select images for training or testing
    #if (img_idx) % 6 != 0:
    file_name = img_fn.split('/')[-1].split('_')[0]
    num_name = int(img_fn.split('/')[-1].split('_')[-1].split('.')[0])
    if num_name % 5 == 0:
        #txt_files_A.append(img_fn)
        if 'Actin' in file_name:
            txt_files_A.append(img_fn)
        elif 'Merged' in file_name:
            txt_files_M.append(img_fn)
        elif 'Nucleus' in file_name:
            txt_files_N.append(img_fn)

with open(txt_img_A, "w") as f:
    for img_fn in txt_files_A:
        f.write(img_fn+"\n")

with open(txt_img_M, "w") as f:
    for img_fn in txt_files_M:
        f.write(img_fn+"\n")

with open(txt_img_N, "w") as f:
    for img_fn in txt_files_N:
        f.write(img_fn+"\n")

        # could we do some uniform sampling of the images so that the training and testing dataset are not biased?
        # in our case there are 980 images in testing and 3920 images in training
        # what if we take one image in every 5 images (4900/980 = 5)?
        #if the reminder of img_idx/5 = 0, assign that image to testing
        #if the reminder of img_idx/5 != 0, assign to training