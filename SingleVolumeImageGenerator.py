# Import the necessary libraries
# For more information about the proposed Single Volume Image Generator please Visit here "https://nilearn.github.io/modules/generated/nilearn.plotting.plot_glass_brain.html"

from nilearn import image
from nilearn import datasets
from nilearn.image import iter_img
from nilearn.plotting import plot_glass_brain
from nilearn import plotting
from matplotlib import pyplot as plt
import os, shutil

#datasets.fetch_abide_pcp(data_dir='C:/Users/zehra/Desktop/UMRAM', n_subjects=20, pipeline='cpac', band_pass_filtering=True, global_signal_regression=True, derivatives=['func_preproc'])

k=2
base_dir = 'C:/Users/zehra/Desktop/UMRAM/ABIDE_pcp/cpac/filt_global' # Path to the Original Data Directory which includes 4D fMRI images of ASD and TC
dump_dir = 'C:/Users/zehra/Desktop/UMRAM/ABIDE_pcp/Data/glass_brain_images'
for filename in os.listdir(base_dir):
    if filename.endswith(".nii"):
        print(filename)
        k=k+1
        my_path = os.path.join(dump_dir, 'preproc' + str(k) + '.nii') # Make a new folder for each subject
        if not os.path.exists(my_path):
            os.mkdir(my_path)
        # Import the subject image
        # Function to consider the 4D fMRI image of the subject for generating glass_brain images 
        rsn = os.path.join(base_dir, filename)
        for i, img in enumerate(iter_img(rsn)):
            plotting.plot_glass_brain(img, threshold=3, display_mode="z",
                              cut_coords=1, colorbar=False)
            plt.savefig(os.path.join(my_path, 'asd%d' % i + ".png"))    
        continue
    else:
        continue

    
    
