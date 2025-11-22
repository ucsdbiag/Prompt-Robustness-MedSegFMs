import os
import glob

org_filenames = sorted(glob.glob('/playpen/soumitri/SAM-Med3D/data_preproc/BTCV_train/*/BTCV/*/*.nii.gz'))

for f in org_filenames:
    imgname = f.split('/')[-1]
    new_imgname = imgname.replace('img', '').replace('label', '')
    new_filename = f.replace(imgname, new_imgname)
    # print(new_filename)
    os.rename(f, new_filename)
