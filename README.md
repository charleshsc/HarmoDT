<p align="center" width="100%">
</p>

<div id="top" align="center">

HarmoDT: Harmony Multi-Task Decision Transformer for Offline Reinforcement Learning
-----------------------------
<img src="https://img.shields.io/badge/Version-1.0.0-blue.svg" alt="Version"> 
<img src="https://img.shields.io/badge/License-Apache_2.0-green.svg" alt="License">

<h4> |<a href="https://arxiv.org/abs/2405.17098"> 📑 Paper </a> |
<a href="https://github.com/charleshsc/QT"> 🐱 Github Repo </a> |
</h4>

<!-- **Authors:** -->

_**Shengchao Hu<sup>1,2</sup>, Ziqing Fan<sup>1,2</sup>, Li Shen<sup>3,4\*</sup>, Ya Zhang<sup>1,2</sup>, Yanfeng Wang<sup>1,2</sup>, Dacheng Tao<sup>5</sup>**_


<!-- **Affiliations:** -->


_<sup>1</sup> Shanghai Jiao Tong University,
<sup>2</sup> Shanghai AI Laboratory,
<sup>3</sup> Sun Yat-sen University,
<sup>4</sup> JD Explore Academy,
<sup>5</sup> Nanyang Technological University._

</div>


## Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Citation](#citation)
- [Acknowledgements](#acknowledgments)


## Overview

The purpose of offline multi-task reinforcement learning (MTRL) is to develop a unified policy applicable to diverse tasks without the need for online environmental interaction. Recent advancements approach this through sequence modeling, leveraging the Transformer architecture's scalability and the benefits of parameter sharing to exploit task similarities. However, variations in task content and complexity pose significant challenges in policy formulation, necessitating judicious parameter sharing and management of conflicting gradients for optimal policy performance.

In this work, we introduce the Harmony Multi-Task Decision Transformer (HarmoDT), a novel solution designed to identify an optimal harmony subspace of parameters for each task. We approach this as a bi-level optimization problem, employing a meta-learning framework that leverages gradient-based techniques. The upper level of this framework is dedicated to learning a task-specific mask that delineates the harmony subspace, while the inner level focuses on updating parameters to enhance the overall performance of the unified policy. Empirical evaluations on a series of benchmarks demonstrate the superiority of HarmoDT, verifying the effectiveness of our approach.



## Quick Start

Download the dataset MT50 via this [Google Drive link](https://drive.google.com/drive/folders/1Ce11F4C6ZtmEoVUzpzoZLox4noWcxCEb).

When your environment is ready, you could run the following script:
``` Bash
python main.py --seed 123 --data_path ./MT50 --prefix_name MT5 # MT30, MT50
```


## Citation
If you find this work is relevant with your research or applications, please feel free to cite our work!
```
@inproceedings{HarmoDT,
    title={HarmoDT: Harmony Multi-Task Decision Transformer for Offline Reinforcement Learning},
    author={Hu, Shengchao and Fan, Ziqing and Shen, Li and Zhang, Ya and Wang, Yanfeng and Tao, Dacheng},
    booktitle={International Conference on Machine Learning},
    year={2024},
}
```

## Acknowledgments

This repo benefits from [DT](https://github.com/kzl/decision-transformer) and [Diffusion-QL](https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL). Thanks for their wonderful works!