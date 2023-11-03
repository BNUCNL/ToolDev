# Ciftify: from the volume space to surface space
# ciftify_recon_all
# Remember to change ciftify_recon_all code before running!!!!
# Line 495: if 'v6.' in fs_version:
# Line 594: add '-nc' between T1w_nii and freesurfer_mgz
export SUBJECTS_DIR=/nfs/z1/zhenlab/BrainImageNet/action/data/bold/derivatives/freesurfer 
export CIFTIFY_WORKDIR=/nfs/z1/zhenlab/BrainImageNet/action/data/bold/derivatives/ciftify/
python run_cmd.py /nfs/z1/zhenlab/BrainImageNet/action/ -c "ciftify_recon_all --surf-reg MSMSulc --resample-to-T1w32k sub-<subject>" -s 08 04 05 06 09 01

# ciftify subject fmri: process the volume data in surface space. This will make output in MNINonLinear Results folder
# if ciftify_recon_all doesn't change. This will make error output
python ciftify_subject_fmri.py
