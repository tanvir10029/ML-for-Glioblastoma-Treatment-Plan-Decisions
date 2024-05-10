# ML-for-Glioblastoma-Treatment-Plan-Decisions

## Step 3: 3D Brain Reconstruction (ML For Health Step 2.ipynb)

1. You will have to use all the files under the folder 3D brain reconstruction to run this notebook
2. In order to download data and run the code you will have to upload the kaggle-2.json file from the folder 3D brain reconstruction
3. For filtering the data according to clinical relevence the file "train_labels.csv" must be used from 3D brain reconstruction
4. For prediction load the model weights "3d_unet_model-2-2.h5" from the folder 3D brain reconstruction

## Step 3: Tumor Localisation and Metrics (ML For Health Step 3.ipynb)
1. Before running the notebook, install FSleyes and ensure that FSL is installed in your environment
2. After FSleyes is installed, add the plugin file (in the plugin folder) as a plugin for FSLeyes through File -> Load plugin.
    * **Ensure that this plugin is in the FSLeyes plugin folder in your local**
3. The t1 and segmentation masks for the patient are under the data folder (Only one patient data was provided due to Github size constraints)
4. While running the notebook, FLIRT registration will take around ~3-4 minutes to finish runnning

Note: The NIFTI files converted to the MNI152 space are already placed in the data folder, just in case of any issues but the output from the notebook should replace these files
   
