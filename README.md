# PA-SfM: Tracker-free differentiable acoustic radiation for freehand 3D photoacoustic imaging    

[***Preprint paper***](https://doi.org/10.13140/RG.2.2.32941.04328)

Three-dimensional (3D) handheld photoacoustic tomography typically relies on bulky and expensive external positioning trackers to correct motion artifacts, which severely limits its clinical flexibility and accessibility. To address this challenge, we present PA-SfM, a tracker-free framework that leverages exclusively single-modality photoacoustic data for both sensor pose recovery and high-fidelity 3D reconstruction via differentiable acoustic radiation modeling. Unlike traditional Structure-from-Motion (SfM) methods that formulate pose estimation as a geometry-driven optimization over visual features, PA-SfM integrates the acoustic wave equation into a differentiable programming pipeline. By leveraging a high-performance, GPU-accelerated acoustic radiation kernel, the framework simultaneously optimizes the 3D photoacoustic source distribution and the sensor array pose via gradient descent. To ensure robust convergence in freehand scenarios, we introduce a coarse-to-fine optimization strategy that incorporates geometric consistency checks and rigid-body constraints to eliminate motion outliers. We validated the proposed method through both numerical simulations and in-vivo rat experiments. The results demonstrate that PA-SfM achieves sub-millimeter positioning accuracy and restores high-resolution 3D vascular structures comparable to ground-truth benchmarks, offering a low-cost, software-defined solution for clinical freehand photoacoustic imaging.

![image](https://github.com/JaegerCQ/PA-SFM/blob/main/pipeline_final.png)        
_The overview of PA-SfM pipeline._    


## BibTeX

```
@article{li2026pa-sfm,    
  title={PA-SfM: Tracker-free differentiable acoustic radiation for freehand 3D photoacoustic imaging},    
  author={Li, Shuang and Gao, Jian and Kim, Chulhong and Choi, Seongwook and Chen, Qian and Wu, Shuang and Wang, Yibing and Zhang, Yu and Huang, Tingting and Zhou, Yucheng and Yao, Boxin and Yao, Yao and Li, Changhui},    
  journal={ResearchGate preprint RG.2.2.32941.04328},    
  year={2026},       
}
```
