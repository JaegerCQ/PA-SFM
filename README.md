# PA-SfM: Tracker-free differentiable acoustic radiation for freehand 3D photoacoustic imaging    

[***Preprint paper***](https://www.biorxiv.org/content/10.64898/2026.04.06.716718v4)

We introduce PA-SfM, a tracker-free differentiable acoustic structure-from-motion (SfM) framework that recovers relative imaging poses directly from PA measurements. By integrating a differentiable acoustic radiation model with hierarchical optimization and rigid array constraints, PA-SfM jointly estimates inter-view transformations and reconstructs 3D PA volumes without external pose measurements. We demonstrate genuine freehand 3D PAI of human hand vasculature, in which arbitrary hand motion over approximately 1 s provides multi-view measurements from which PA-SfM recovers the relative poses and jointly reconstructs a large FOV vascular network without motion tracking or predefined trajectories.  

![image](https://github.com/JaegerCQ/PA-SfM/tree/LS-GJ/pictures/sequential_display.png)           
_Repeatability validation of PA-SfM freehand 3D reconstruction of hand vessels._        

![image](https://github.com/JaegerCQ/PA-SfM/tree/LS-GJ/pictures/pipeline_final.png)        
_The overview of PA-SfM pipeline._    

![image](https://github.com/JaegerCQ/PA-SfM/tree/LS-GJ/pictures/freehand.png)        
_PA-SfM freehand 3D reconstructions of hand vessels._

## Create Conda Environment   

To ensure reproducible results, it is strongly recommended to use the following pinned installation configuration and run the experiments on NVIDIA RTX 4090D with CUDA 12.6.   
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

Note: If you need to use this data in any context, please make sure to contact junhuishi@outlook.com.
Place input files under `data/` with names expected by `run_group3_pose_range.sh`, for example:

```text
data/
  sensor_location_group03_pose000.txt
  processed_signal_group03_pose000.txt
  processed_signal_group03_pose001.txt
  ...
```

## Settings

Adjust this according to the number of GPUs you have.
In `run_group3_pose_range.sh`, modify:

```shellscript
GPU_IDS=(0 1 2 3)
```

The default setup is intended to run on four RTX 4090D GPUs. Each pose reconstruction takes about 26 minutes, and the full PA-SfM run for all 10 poses completes in under 4 hours.

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

## Citation 

```   
@article{li2026pa,   
  title={PA-SfM: Tracker-free differentiable acoustic radiation for freehand 3D photoacoustic imaging},        
  author={Li, Shuang and Gao, Jian and Kim, Chulhong and Choi, Seongwook and Huang, Hao and Wang, Xuanhao and Shi, Junhui and Chen, Qian and Wang, Yibing and Wu, Shuang and Zhang, Yu and Huang, Tingting and Zhou, Yucheng and Yao, Boxin and Yao, Yao and Li, Changhui},      
  journal={bioRxiv},       
  pages={2026--04},       
  year={2026},      
  publisher={Cold Spring Harbor Laboratory}  
}    
```

```
@article{wang2025cross,  
  title={Cross-regional real-time visualization of systemic physiology and dynamics with 3D panoramic photoacoustic computed tomography (3D-PanoPACT)},  
  author={Wang, Xuanhao and Meng, Yuqian and Sun, Mingli and Gao, Xiali and Wang, Yuqi and Wang, Shaobo and Wang, Kaiyue and Wang, Ruofan and Ren, Danyang and Yin, Yonggang and others},  
  journal={Nature Communications},  
  volume={16},  
  number={1},  
  pages={10077},  
  year={2025},  
  publisher={Nature Publishing Group UK London}  
}  
```

```
@article{choi2023deep,
  title={Deep learning enhances multiparametric dynamic volumetric photoacoustic computed tomography in vivo (DL-PACT)},
  author={Choi, Seongwook and Yang, Jinge and Lee, Soo Young and Kim, Jiwoong and Lee, Jihye and Kim, Won Jong and Lee, Seungchul and Kim, Chulhong},
  journal={Advanced Science},
  volume={10},
  number={1},
  pages={2202089},
  year={2023},
  publisher={Wiley Online Library}
}
```

## Ackonwledgement

We are deeply grateful to Professor Chulhong Kim, Professor Junhui Shi, Dr. Seongwook Choi, Dr. Xuanhao Wang, Dr. Hao Huang and Dr. Zhibo Xiao for providing the invaluable in vivo experimental data.
