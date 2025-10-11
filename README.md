<p align="center">
  <h1 align="center">STAR-Bench: Probing Deep Spatio-Temporal Reasoning as Audio 4D Intelligence</h1>
    <p align="center">
    <a href="https://scholar.google.com/citations?user=iELd-Q0AAAAJ"><strong>Zihan Liu</strong></a>
    ·
    <a href=""><strong>Zhikang Niu</strong></a>
    ·
    <a href=""><strong>Qiuyang Xiao</strong></a>
    ·
    <a href=""><strong>Zhisheng Zheng</strong></a>
    ·
    <a href=""><strong>Ruoqi Yuan</strong></a>
    ·
    <a href="https://yuhangzang.github.io/"><strong>Yuhang Zang</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=sJkqsqkAAAAJ"><strong>Yuhang Cao</strong></a>
    ·
    <a href="https://lightdxy.github.io/"><strong>Xiaoyi Dong</strong></a>
    ·
    <a href="https://lightdxy.github.io/"><strong>Jianze Liang</strong></a>
    ·
    <a href="https://lightdxy.github.io/"><strong>Xie Chen</strong></a>
    ·
     <a href="http://dahua.site/"><strong>Leilei Sun</strong></a>
    ·
     <a href="http://dahua.site/"><strong>Dahua Lin</strong></a>
    ·
     <a href="https://myownskyw7.github.io/"><strong>Jiaqi Wang</strong></a>
  </p>

  📖<a href="">Paper</a> |
 | 🤗<a href="https://huggingface.co/datasets/internlm/Spark-Data">STAR-Bench Dataset</a></h3> | 
<div align="center"></div>
<p align="center">
  <p>

## 📢 News
<!-- - 🚀 [09/29/2025] We release our 🤗<a href="https://huggingface.co/datasets/internlm/Spark-Data">datasets</a>.
- 🚀 [09/29/2025] We release our **Spark's** <a href="https://arxiv.org/abs/2509.22624">Paper</a>.
- 🚀 [09/29/2025] We upload our evaluation code and model checkpoints. -->
- 🚀 [09/30/2025] We release **STAR-Bench** repository.


## 🌈Overview of STAR-Bench
Despite rapid progress in Multi-modal Large Language Models and Large Audio-Language Models, existing audio benchmarks largely test semantics that can be recovered from text captions, masking deficits in fine-grained perceptual reasoning.
We formalize <strong>audio 4D intelligence</strong> that is defined as reasoning over sound dynamics in time and 3D space, and introduce a <strong>STAR-Bench</strong> to measure it. STAR-Bench combines a <strong>Foundational Acoustic Perception</strong>setting (six attributes under absolute and relative regimes) with a <strong>Holistic Spatio-Temporal Reasoning</strong> setting that includes segment reordering for continuous and discrete processes and spatial tasks spanning static localization, multi-source relations, and dynamic trajectories.
Unlike prior benchmarks where caption-only answering reduces accuracy slightly, STAR-Bench induces far larger drops (-31.5\% temporal, -35.2\% spatial), evidencing its focus on <strong>linguistically hard-to-describe cues</strong>. Evaluating 19 models reveals substantial gaps to humans and a capability hierarchy: closed-source models are bottlenecked by fine-grained perception, while open-source models lag across perception, knowledge, and reasoning. Our STAR-Bench provides critical insights and a clear path forward for developing future models with a more robust understanding of the physical world.
</p>
    <a href="">
      <img src="assets/bench_examples.png" alt="Logo" width="100%"> 
    </a>
<br>


## DATA CURATION
<a href="">
      <img src="assets/data_dist.png" alt="Logo" width="90%"> 
</a>
Our data curation pipeline integrates procedural synthesis with real-world data collection to ensure both comprehensive coverage and ecological validity. All audio for the foundational perception task is synthesized using precise parameterization or the Pyroomacoustics physics-based simulator, providing complete control over acoustic parameters. Domain experts rigorously validate the task difficulty
levels, which are then calibrated through human testing. For the holistic spatio-temporal reasoning task, the curation process comprises four key stages, including human annotation and final selection based on human performance.
<a href="">
      <img src="assets/pipeline.png" alt="Logo" width="90%"> 
</a>



## Benchmark Results



Evaluating 19 models reveals substantial gaps to humans and a capability hierarchy: closed-source models are bottlenecked by fine-grained
perception, while open-source models lag across perception, knowledge, and reasoning. 

## 💡 Several key insights for the future development of open-source audio model
- 🔥 **Enhancing dense audio captioning.**: Open-source models struggle to produce dense, fine-grained captions, which limits their perceptual sensitivity and ability to extract embedded knowledge. Bridging this gap is a crucial first step. 
- 🔥 **Improving multi-audio reasoning.**: Open-source models lag significantly in comparing, integrating, and grounding information across multiple audio clips. 
- 🔥 **Moving beyond channel-averaged audio preprocessing.**: The common practice of averaging multi-channel audio into a mono signal is a major bottleneck for spatial reasoning. Developing architectures that natively process multi-channel cues is essential for unlocking genuine spatial awareness.

## ⚙️ Evaluation











## 🛠️ Setup
```
git clone https://github.com/InternLM/Spark.git
conda create -n Lmm_xc python=3.10
conda activate Visual-RFT
cd /Spark/Lmm_XC
pip install -e .[vllm]
pip install flash_attn --no-build-isolation
```
Lmm_XC is developed upon modifications to the LMM-R1 project, and its installation process can be referred to the LMM-R1 instructions.

## Datasets
🔦 Our dataset includes the training data for **Spark-VL-7B** and **Spark-VL-32B** models, as well as a collection of all **multimodal mathematical benchmarks**. It can be directly downloaded and used. Refer to 🤗<a href="https://huggingface.co/datasets/internlm/Spark-Data">datasets</a>.

## Inference
We have uploaded the model <strong>Spark-VL-7B</strong> (<a href="https://huggingface.co/internlm/Spark-VL-7B">🤗Huggingface</a>). You can use it to evaluate the inference performance of on Multimodal Mathematical Benchmarks and Reward-Related Benchmarks. 
It should be noted that during our training process, we append the following prompt at the end of the input to facilitate answer extraction. Therefore, it is recommended to also append this prompt at the end during testing.
```
 Please first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.
```





## ✒️Citation
```
TBD
```

## 📄 License
![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg) ![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg) **Usage and License Notices**: The data and code are intended and licensed for research use only.
License: Attribution-NonCommercial 4.0 International It should abide by the policy of OpenAI: https://openai.com/policies/terms-of-use

## Acknowledgement
We sincerely thank projects <a href="https://github.com/TideDra/lmm-r1">lmm-r1</a> and <a href="https://github.com/OpenRLHF/OpenRLHF">OpenRLHF</a> for providing their open-source resources.






