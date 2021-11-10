# Import the necessary libraries
# For more information about the proposed Single Volume Image Generator please Visit here "https://nilearn.github.io/modules/generated/nilearn.plotting.plot_glass_brain.html"

from nilearn import image
from nilearn import datasets
from nilearn.image import iter_img
from nilearn.plotting import plot_glass_brain
from nilearn.plotting import plot_stat_map
from nilearn import plotting
from matplotlib import pyplot as plt
import os, shutil

#datasets.fetch_abide_pcp(data_dir='C:/Users/zehra/Desktop/UMRAM', n_subjects=10, pipeline='cpac', band_pass_filtering=True, global_signal_regression=True, derivatives=['func_preproc'])

k=0
base_dir = 'C:/Users/zehra/Desktop/UMRAM/ABIDE_pcp/cpac/filt_global' # Path to the Original Data Directory which includes 4D fMRI images of ASD and TC
glass_dir = 'C:/Users/zehra/Desktop/UMRAM/ABIDE_pcp/Data/glass_brain_images'
stat_dir = 'C:/Users/zehra/Desktop/UMRAM/ABIDE_pcp/Data/stat_images'

for filename in os.listdir(base_dir):
    if filename.endswith(".nii"):
        print(filename)
        k=k+1
        glass_path = os.path.join(glass_dir, 'preproc_glass' + str(k) + '.nii') # Make a new folder for each subject
        stat_path = os.path.join(stat_dir, 'preproc_stat' + str(k) + '.nii')
        if not os.path.exists(glass_path):
            os.mkdir(glass_path)
        if not os.path.exists(stat_path):
            os.mkdir(stat_path) 
        # Import the subject image
        # Function to consider the 4D fMRI image of the subject for generating glass_brain images 
        rsn = os.path.join(base_dir, filename)
        for i, img in enumerate(iter_img(rsn)):
            plotting.plot_glass_brain(img, threshold=3, display_mode="z",
                              cut_coords=1, colorbar=False)
            plt.savefig(os.path.join(glass_path, 'asd%d' % i + ".png"))
            plt.close('all')

        for i, img in enumerate(iter_img(rsn)):
            plotting.plot_stat_map(img, threshold=5, display_mode="z",
                              cut_coords=1, colorbar=False)
            plt.savefig(os.path.join(stat_path, 'asd%d' % i + ".png"))   
            plt.close('all')  
        continue
    else:
        continue

    
    
