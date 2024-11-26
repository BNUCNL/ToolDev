# BIDS Data structure transformation
python data2bids.py /nfs/z1/zhenlab/BrainImageNet/action scaninfo.xlsx -q ok --overwrite --skip-feature-validation  

# fMRIprep preprocessing 
export prjdir=/nfs/z1/zhenlab/BrainImageNet/action/data/bold
export bids_fold=$prjdir/nifti
export out_dir=$prjdir/derivatives
export work_dir=$prjdir/workdir
export license_file=/usr/local/neurosoft/freesurfer/license.txt

fmriprep-docker $bids_fold $out_dir participant --skip-bids-validation --participant-label 01 02 03 04 05 06 07 08 09 10 12 13 15 19 20 28 29 30 --fs-license-file $license_file --output-spaces anat fsnative -w $work_dir
