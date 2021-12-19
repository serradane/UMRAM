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
#dx_group
#datasets.fetch_abide_pcp(data_dir = 'C:/Users/zehra/Desktop/UMRAM', n_subjects = 10, pipeline = 'cpac', band_pass_filtering =True, global_signal_regression=True, derivatives=['func_preproc'], DX_GROUP=2)

k=0
#base_dir_asd = 'C:/Users/zehra/Desktop/UMRAM/ABIDE_pcp/cpac/filt_global' # Path to the Original Data Directory which includes 4D fMRI images of ASD
#glass_dir_asd = 'C:/Users/zehra/Desktop/UMRAM/ABIDE_pcp/Data/glass_brain_images'
#stat_dir_asd = 'C:/Users/zehra/Desktop/UMRAM/ABIDE_pcp/Data/stat_images'
base_dir_control = 'C:/Users/zehra/Desktop/UMRAM/ABIDE_pcp/cpac/filt_global/Control' # Path to the Original Data Directory which includes 4D fMRI images of TC
glass_dir_control = 'C:/Users/zehra/Desktop/UMRAM/ABIDE_pcp/Data/glass_brain_control'
stat_dir_control = 'C:/Users/zehra/Desktop/UMRAM/ABIDE_pcp/Data/stat_control'

# for filename in os.listdir(base_dir_asd):
#     if filename.endswith(".nii"):
#         print(filename)
#         k=k+1
#         glass_path_asd = os.path.join(glass_dir_asd, filename) # Make a new folder for each subject
#         stat_path_asd = os.path.join(stat_dir_asd, filename)
#         if not os.path.exists(glass_path_asd):
#             os.mkdir(glass_path_asd)
#         if not os.path.exists(stat_path_asd):
#             os.mkdir(stat_path_asd) 
#         # Import the subject image
#         # Function to consider the 4D fMRI image of the subject for generating glass_brain images 
#         rsn = os.path.join(base_dir_asd, filename)
#         for i, img in enumerate(iter_img(rsn)):
#             plotting.plot_glass_brain(img, threshold=3, display_mode="z",
#                               cut_coords=1, colorbar=False)
#             plt.savefig(os.path.join(glass_path_asd, 'asd%d' % i + ".png"))
#             plt.close('all')

#         for i, img in enumerate(iter_img(rsn)):
#             plotting.plot_stat_map(img, threshold=5, display_mode="z",
#                               cut_coords=1, colorbar=False)
#             plt.savefig(os.path.join(stat_path_asd, 'asd%d' % i + ".png"))   
#             plt.close('all')  
#         continue
#     else:
#         continue

for filename in os.listdir(base_dir_control):
    if filename.endswith(".nii"):
        print(filename)
        k=k+1
        glass_path_control = os.path.join(glass_dir_control, filename) # Make a new folder for each subject
        stat_path_control = os.path.join(stat_dir_control, filename)
        if not os.path.exists(glass_path_control):
            os.mkdir(glass_path_control)
        if not os.path.exists(stat_path_control):
            os.mkdir(stat_path_control) 
        # Import the subject image
        # Function to consider the 4D fMRI image of the subject for generating glass_brain images 
        rsn = os.path.join(base_dir_control, filename)
        for i, img in enumerate(iter_img(rsn)):
            plotting.plot_glass_brain(img, threshold=3, display_mode="z",
                              cut_coords=1, colorbar=False)
            plt.savefig(os.path.join(glass_path_control, 'control%d' % i + ".png"))
            plt.close('all')

        for i, img in enumerate(iter_img(rsn)):
            plotting.plot_stat_map(img, threshold=5, display_mode="z",
                              cut_coords=1, colorbar=False)
            plt.savefig(os.path.join(stat_path_control, 'control%d' % i + ".png"))   
            plt.close('all')  
        continue
    else:
        continue

    
    
