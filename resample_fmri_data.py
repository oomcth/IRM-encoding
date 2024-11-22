import numpy as np
import glob
import os
from tqdm.auto import tqdm
from nilearn.image import resample_img

import llms_brain_lateralization as lbl
from llms_brain_lateralization import make_dir

target_affine = np.array([[   4.,    0.,    0.,  -72.],
                          [   0.,    4.,    0., -106.],
                          [   0.,    0.,    4.,  -64.],
                          [   0.,    0.,    0.,    1.]])

target_shape = (37, 46, 38)

output_path = lbl.fmri_data_resampled

subject_list = np.sort(glob.glob(os.path.join(lbl.fmri_data, 'sub-EN*')))

for sub_id in tqdm(subject_list):
    sub_name = os.path.basename(sub_id)
    make_dir(os.path.join(output_path, sub_name))
    fmri_imgs_sub = sorted(glob.glob(os.path.join(sub_id, 'func/*.nii.gz')))
    for run, fmri_imgs_sub_run in enumerate(fmri_imgs_sub):
        img_resampled = resample_img(fmri_imgs_sub_run, 
                                     target_affine=target_affine, 
                                     target_shape=target_shape)
        img_resampled.to_filename(os.path.join(output_path,
                                               sub_name,
                                               '{}_run{}.nii.gz'.format(sub_name, run+1)))