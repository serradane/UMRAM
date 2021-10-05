# Import the necessary libraries
# For more information about the proposed Single Volume Image Generator please Visit here "https://nilearn.github.io/modules/generated/nilearn.plotting.plot_glass_brain.html"

from nilearn import image
from nilearn import datasets
from nilearn.image import iter_img
from nilearn.plotting import plot_glass_brain
from nilearn import plotting
from matplotlib import pyplot as plt
import os, shutil

#TODO Bu pathi degistirmen gerekir
datasets.fetch_abide_pcp(data_dir='C:/Users/zehra/Desktop/UMRAM', n_subjects=1, pipeline='cpac', band_pass_filtering=True, global_signal_regression=True, derivatives=['func_preproc'])

original_dir = 'C:/Users/zehra/Desktop/UMRAM/ABIDE_pcp/cpac/filt_global' # Path to the Original Data Directory which includes 4D fMRI images of ASD and TC
my_path = os.path.join(original_dir, 'MaxMun_d_0051355_func_preproc.nii') # Make a new folder for each subject
os.mkdir(my_path)
# rsn = 'C:/Users/zehra/Desktop/UMRAM/ABIDE_pcp/cpac/filt_global/MaxMun_d_0051355_func_preproc.nii' # Import the subject image
rsn = 'C:/Users/zehra/Desktop/UMRAM/ABIDE_pcp/cpac/filt_global/Pitt_0050003_func_preproc.nii.gz'
# Function to consider the 4D fMRI image of the subject for generating glass_brain images 
for i, img in enumerate(iter_img(rsn)):
    plotting.plot_glass_brain(img, threshold=3, display_mode="z",
                              cut_coords=1, colorbar=False)
    plt.savefig(os.path.join(my_path, 'asd%d' % i + ".png"))
    
    
