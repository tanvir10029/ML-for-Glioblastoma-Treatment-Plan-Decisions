# ML-for-Glioblastoma-Treatment-Plan-Decisions

## Step 3: Tumor Localisation and Metrics
1. Before running the notebook, install FSleyes and ensure that FSL is installed in your environment
2. After FSleyes is installed, add the plugin file (in the plugin folder) as a plugin for FSLeyes through File -> Load plugin.
    * Ensure that this plugin is in the FSLeyes plugin path
3. The t1 and segmentation masks for the patient are under the data folder (Only one patient data was provided due to Github size constraints)
4. While running the notebook, FLIRT registration will take around ~3-4 minutes to finish runnning

Note: The NIFTI files converted to the MNI152 space are already placed in the data folder, just in case of any issues but the output from the notebook should replace these files
   
