import argparse
import os
import glob
import sys

parser = argparse.ArgumentParser(description='datasets list')
# modify this to point to the folder
parser.add_argument('--img_path', type=str, default='/media/jyo/9CAE3358AE332A62/Project_NTU/MCF-7')
parser.add_argument('--out_txt_prefix', default='/home/jyo/code/SUTD_Cell/datasets/cell_MCF')
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
    if img_idx > int(num_img * 0.8):
        file_name = img_fn.split('/')[-1].split('_')[0]
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

