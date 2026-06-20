# PA-SfM: Tracker-free differentiable acoustic radiation for freehand 3D photoacoustic imaging    

[***Preprint paper***](https://www.biorxiv.org/content/10.64898/2026.04.06.716718v4)

We introduce PA-SfM, a tracker-free differentiable acoustic structure-from-motion (SfM) framework that recovers relative imaging poses directly from PA measurements. By integrating a differentiable acoustic radiation model with hierarchical optimization and rigid array constraints, PA-SfM jointly estimates inter-view transformations and reconstructs 3D PA volumes without external pose measurements. We demonstrate genuine freehand 3D PAI of human hand vasculature, in which arbitrary hand motion over approximately 1 s provides multi-view measurements from which PA-SfM recovers the relative poses and jointly reconstructs a large FOV vascular network without motion tracking or predefined trajectories.  

![image](https://github.com/JaegerCQ/PA-SfM/tree/LS-GJ/pictures/sequential_display.png)           
_Repeatability validation of PA-SfM freehand 3D reconstruction of hand vessels._       

## Create Conda Environment   

To ensure reproducible results, it is strongly recommended to use the following pinned installation configuration and run the experiments on an NVIDIA RTX 4090D with CUDA 12.6.   
```bash
conda create -n PA_SfM --file locks/conda-explicit.txt
conda activate PA_SfM

python -m pip install \
  --no-index \
  --find-links locks/wheelhouse \
  --require-hashes \
  -r locks/requirements.pip-hash-lock.txt
```

If exact one-to-one reproducibility is not required, you can also create the environment using the following method.

```bash
conda create -n PA_SfM python=3.11 -y
conda activate PA_SfM
conda install -c conda-forge numpy scipy matplotlib jupyterlab ipykernel -y
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu126
```

## Data Layout   

Place input files under `data/` with names expected by `run_group3_pose_range.sh`, for example:

```text
data/
  sensor_location_group03_pose000.txt
  processed_signal_group03_pose000.txt
  processed_signal_group03_pose001.txt
  ...
```

## Run Pipeline  

```bash
conda activate PA_SfM
chmod +x run_group3_pose_range.sh
nohup ./run_group3_pose_range.sh > main_group3_pose_range.log 2>&1 &
```

Monitor progress:

```bash
tail -f main_group3_pose_range.log
```

## BibTeX

```   
@article{li2026pa,   
  title={PA-SfM: Tracker-free differentiable acoustic radiation for freehand 3D photoacoustic imaging},        
  author={Li, Shuang and Gao, Jian and Kim, Chulhong and Choi, Seongwook and Huang, Hao and Wang, Xuanhao and Shi, Junhui and Chen, Qian and Wang, Yibing and Wu, Shuang and Zhang, Yu and Huang, Tingting and Zhou, Yucheng and others},      
  journal={bioRxiv},       
  pages={2026--04},       
  year={2026},      
  publisher={Cold Spring Harbor Laboratory}  
}    
```
