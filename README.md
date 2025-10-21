<p align="center">
  <h1 align="center">
    <div style="display: flex; align-items: center; justify-content: center;">
      <img src="assets/4d_logo.png" alt="logo" height="100" style="margin-right: 12px;">
      <div style="text-align: left; line-height: 1.3;">
        STAR-Bench: Probing Deep Spatio-Temporal Reasoning as Audio 4D Intelligence
      </div>
    </div>
  </h1>
    <p align="center">
    <a href="https://scholar.google.com/citations?user=iELd-Q0AAAAJ"><strong>Zihan Liu<sup>*</sup></strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=mXSpi2kAAAAJ&hl=zh-CN"><strong>Zhikang Niu<sup>*</sup></strong></a>
    ·
    <a href="https://github.com/akkkkkkkkki/"><strong>Qiuyang Xiao</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=WYwBrzAAAAAJ&hl=en"><strong>Zhisheng Zheng</strong></a>
    ·
    <a href="https://github.com/yrqqqq404"><strong>Ruoqi Yuan</strong></a>
    ·
    <a href="https://yuhangzang.github.io/"><strong>Yuhang Zang<sup>&dagger;</sup></strong></a>
    </br>
    <a href="https://scholar.google.com/citations?user=sJkqsqkAAAAJ"><strong>Yuhang Cao</strong></a>
    ·
    <a href="https://lightdxy.github.io/"><strong>Xiaoyi Dong</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=P4yNnSkAAAAJ&hl=zh-TW"><strong>Jianze Liang</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=d6u01FkAAAAJ&hl=en"><strong>Xie Chen</strong></a>
    ·
     <a href="https://scholar.google.com/citations?user=QVHvhM4AAAAJ&hl=en"><strong>Leilei Sun</strong></a>
    ·
     <a href="http://dahua.site/"><strong>Dahua Lin</strong></a>
    ·
     <a href="https://myownskyw7.github.io/"><strong>Jiaqi Wang<sup>&dagger;</sup></strong></a>
  </p>
  <p align="center" style="font-size: 1em; margin-top: -1em"> <sup>*</sup>  Equal Contribution. <sup>&dagger;</sup>Corresponding authors. </p>
<p align="center" style="font-size: 1.2em; margin-top: 0.5em">
  📖<a href="">arXiv</a> |
  🌐<a href="">Homepage</a>
 | 🤗<a href="https://huggingface.co/datasets/internlm/Spark-Data">Dataset</a>
</p> 
<div align="center"></div>



## 📢 News
- 🚀 [09/30/2025] We release STAR-Bench repository and homepage.


## 🌈Overview
We formalize <strong>audio 4D intelligence</strong> that is defined as reasoning over sound dynamics in time and 3D space, and introduce a <strong>STAR-Bench</strong> to measure it. STAR-Bench combines a <strong>Foundational Acoustic Perception</strong>setting (six attributes under absolute and relative regimes) with a <strong>Holistic Spatio-Temporal Reasoning</strong> setting that includes segment reordering for continuous and discrete processes and spatial tasks spanning static localization, multi-source relations, and dynamic trajectories.
Unlike prior benchmarks where caption-only answering reduces accuracy slightly, STAR-Bench induces far larger drops (-31.5\% temporal, -35.2\% spatial), evidencing its focus on <strong>linguistically hard-to-describe cues</strong>. Evaluating 19 models reveals substantial gaps to humans and a capability hierarchy. Our STAR-Bench provides critical insights and a clear path forward for developing future models with a more robust understanding of the physical world. Benchmark examples are illustrated below.
</p>
<p style="text-align: center;"> 
  <img src="assets/bench_examples.png" alt="STAR-Bench Examples" width="100%"> 
</p>
  
A comparative overview of our benchmark against other representative audio benchmarks is shown below.
<p style="text-align: center;"> 
<img src="assets/bench_compare.png" alt="Comparison among Benchmarks" width="100%"> 
</p> 






## 📊Results and Analysis
Evaluation results of various models on STAR-Bench:
<p style="text-align: center;">
  <img src="assets/results.png" alt="Results" width="100%">
</p>
Error distribution across temporal and spatial Tasks:
<p style="text-align: center;">
  <img src="assets/error_dist.png" alt="Results" width="100%">
</p>

## 💡 Key Insights
- 🔥**A clear capability hierarchy between the two groups.** Closed-source models are bottlenecked by fine-grained perception, while open-source models lag across perception, knowledge, and reasoning. 
- 🔥 **Enhancing dense audio captioning.** Open-source models struggle to produce dense, fine-grained captions, which limits their perceptual sensitivity and ability to extract embedded knowledge. Bridging this gap is a crucial first step. 
- 🔥 **Improving multi-audio reasoning.** Open-source models lag significantly in comparing, integrating, and grounding information across multiple audio clips. 
- 🔥 **Moving beyond channel-averaged audio preprocessing.** The common practice of averaging multi-channel audio into a mono signal is a major bottleneck for spatial reasoning. Developing architectures that natively process multi-channel cues is essential for unlocking genuine spatial awareness.



## ⚙️Data Curation
<p style="text-align: center;">
<img src="assets/data_dist.png" alt="" width="90%"> 
</p>
 All audio for the foundational perception task is synthesized using precise parameterization or the Pyroomacoustics physics-based simulator, providing complete control over acoustic parameters. Domain experts rigorously validate the task difficulty
levels, which are then calibrated through human testing.</br>
For the holistic spatio-temporal reasoning task, the curation process comprises four key stages, including human annotation and final selection based on human performance, as illustrated below.
<p style="text-align: center;">
  <img src="assets/pipeline.png" alt="pipeline" width="90%"> 
</p>

## 🛠️  Test Your Model ！



## ✒️Citation
```
TBD
```

## 📄 License
![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg) ![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg) **Usage and License Notices**: The data and code are intended and licensed for research use only.

## Acknowledgement
We sincerely thank <a href="https://www.molardata.com/" target="_blank">MolarData</a> for providing the platform that supported our data annotation, verification, and review processes.







