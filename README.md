

<p align="center">

  <h2 align="center">MovieLLM: <br> Enhancing Long Video Understanding with AI-Generated Movies</h2>
  <p align="center">
    <a href="https://github.com/Deaddawn"><strong>Zhende Song</strong></a>
    ·
    <a href="https://github.com/doctorlightt"><strong>Chenchen Wang</strong></a>
    ·  
    <a href="https://github.com/sjmFDU"><strong>Jiamu Sheng</strong></a>
    ·
    <a href="https://icoz69.github.io/"><strong>Chi Zhang†</strong></a>
    ·
    <a href="https://scholar.google.com/citations?hl=zh-CN&user=gsLd2ccAAAAJ"><strong>Jiayuan Fan✦</strong></a>
    ·
    <a href="https://eetchen.github.io/"><strong>Tao Chen</strong></a>
    <br>
    ( † Project Leader, ✦ Corresponding Author )
    <br>
    From Fudan University and Tencent PCG
    <br>
    </br>
        <a href="https://arxiv.org/abs/2403.01422">
        <img src='https://img.shields.io/badge/arxiv-MovieLLM-b31b1b.svg' alt='Paper PDF'></a>
        <a href="https://deaddawn.github.io/MovieLLM/">
        <img src='https://img.shields.io/badge/Project-Website-green' alt='Project Page'></a>
  </p>
</p>




<image src="docs/fig1.png" />
We propose MovieLLM, a novel framework designed to create synthetic, high-quality data for long videos. This framework leverages the power of GPT-4 and text-to-image models to generate detailed scripts and corresponding visuals. 


## Changelog
- __[2024.03.03]__: Release inference code, evaluation code and model weights.


## Summary
This repository is mainly used for these purposes: data generation code, training code, video evaluation code. We build this repo based on LLaMA-VID. We plan to first release our model, inference and evaluation code and then the rest.  
<b>For a better understanding of our training and evaluation process, we suggest running through codes from [LLaMA-VID](https://github.com/dvlab-research/LLaMA-VID) first.</b>

## Contents
- [Install](#install)
- [Model](#model)
- [Preparation](#preparation)
- [MovieLLM pipeline](#pipeline)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Results](#results)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)


## Install
Please follow the instructions below to install the required packages. Our training process is mainly based on LLaMA-VID. And our short video evaluation process is mainly based on quantitative_evaluation from Video-ChatGPT.
1. Clone this repository
```bash
git clone https://github.com/Deaddawn/MovieLLM-code.git
```
2. Clone LLaMA-VID repository
```bash
cd MovieLLM-code
git clone https://github.com/dvlab-research/LLaMA-VID.git
mv eval_movie_qa.py calculate.py LLaMA-VID
```

3. Install Package
```bash
conda create -n MovieLLM python=3.10 -y
conda activate MovieLLM
cd LLaMA-VID
pip install -e .
```

4. Install additional packages for video training
```bash
pip install ninja
pip install flash-attn --no-build-isolation
```


## Model
We provide our baseline model and model trained on our generated dataset. All models are trained on stage3 of LLaMA-VID. For more detailed information, please refer to [LLaMA-VID-model](https://github.com/dvlab-research/LLaMA-VID#model)

| Type | Max Token | Base LLM | Finetuning Data | Finetuning schedule | Download |
|----------|----------|----------|---------------|--------------------|------------------|
Long video | 64K | Vicuna-7B-v1.5 | LLaVA1.5-VideoChatGPT-Instruct + LongVideoQA | full_ft-1e | [ckpt](https://huggingface.co/sfsdfsafsddsfsdafsa/llama-vid-7b-full-224-long-video-MovieLLM/tree/main) |
Long video | 64K | Vicuna-7B-v1.5 | LLaVA1.5-VideoChatGPT-Instruct + LongVideoQA+MovieLLMQA | full_ft-1e | [ckpt](https://huggingface.co/sfsdfsafsddsfsdafsa/llama-vid-7b-full-224-long-video-MovieLLM/tree/main/llama-vid-7b-full-224-long-video-MovieLLM) |

## Preparation
This section is mainly used to demonstrate how to set up the data and model environment related to llamavid. Again, we suggest running through from the original [LLaMA-VID-preparation](https://github.com/dvlab-research/LLaMA-VID?tab=readme-ov-file#preparation). We write this section based on the above with some alteration.


### Dataset
We provide raw dataset generated from our pipeline and also related training data based on LLaMA-VID.

#### Our Raw Data
Data generated from our pipeline consists of key frame images, corresponding QAs and dialogues. You can download it from here [MovieLLM-Data (coming soon)]()
<image src="docs/tuning_data_distribution.png" />

#### Training Data
To run training process on LLaMA-VID stage-3, processed video data and corresponding QA pairs are needed:  
##### (1) Processed Video Data
We first preprocess the raw data from MovieNet (used in LLaMA-VID original paper) and the raw data generated from our pipeline. 

For data preprocessing from MovieNet, please first download the long video data from [MovieNet](https://movienet.github.io/), shot detection results from [here](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155186668_link_cuhk_edu_hk/EYbaGk86_WNFm9YP45WVQ_oB0GGkusDNBRwQQ19vBy4z2A?e=cKbiHJ). Place shot detection results under `LLaMA-VID-Finetune/movienet/files` before preprocessing. Then please follow the [preprocess-instruct](https://github.com/dvlab-research/LLaMA-VID/blob/main/README.md#long-video) to preprocess your data

For processed data from ours, please download it from here [MovieLLM-feat (coming soon)](). 
##### (2) Corresponding QA Pairs
For correspongding QA pairs, please download it from here:
| Data file name | Size |
| --- | ---: |
| [long_videoqa_base.json](https://huggingface.co/datasets/sfsdfsafsddsfsdafsa/Long-video-test-data/tree/main) | 240MB |
| [long_videoqa_ours.json](https://huggingface.co/datasets/sfsdfsafsddsfsdafsa/Long-video-test-data/tree/main) | 245MB |


### Pretrained Weights
Please download the pretrained weights from the following link [EVA-ViT-G](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth), [QFormer-7b](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/InstructBLIP/instruct_blip_vicuna7b_trimmed.pth)

### Structure

Please organize the video data, QA pairs and weights as the following structure:

```
LLaMA-VID
├── llamavid
├── scripts
├── work_dirs
│   ├── llama-vid
│   │   ├── llama-vid-7b-full-224-long-video-MovieLLM
│   │   ├── llama-vid-7b-full-224-long-video-baseline
├── model_zoo
│   ├── LAVIS
│   │   ├── eva_vit_g.pth
│   │   ├── instruct_blip_vicuna7b_trimmed.pth
├── data
│   ├── LLaMA-VID-Finetune
│   │   ├── long_videoqa_base.json
│   │   ├── long_videoqa_ours.json
│   │   ├── movienet
│   │   ├── story_feat
│   ├── LLaMA-VID-Eval
│   │   ├── MSRVTT-QA
│   │   ├── MSVD-QA
│   │   ├── video-chatgpt
```






## Pipeline
Coming soon.
<image src="docs/PIPELINE.png" />

## Training
Coming soon.

## Inference
For long-video inference on LLaMA-VID, please follow [LLaMA-VID-Long-video-preprocess](https://github.com/dvlab-research/LLaMA-VID/blob/main/README.md#long-video-inference) to process your video.
Then, please try this for long video inference:
```bash
cd LLaMA-VID
python llamavid/serve/run_llamavid_movie.py \
    --model-path work_dirs/llama-vid/llama-vid-7b-full-224-long-video \
    --video-file <path to your processed video file> \
    --load-4bit
```

## Evaluation
We perform evaluation on both short video and long video.
### Short video
For short video evaluation, please download the evaluation data following [Preparation](https://github.com/dvlab-research/LLaMA-VID/blob/main/README.md#preparation) and organize them as in [Structure](#structure).
#### Results for short video
Model | MSVD-QA | MSVD-QA Score | MSRVTT-QA | MSRVTT-QA Score | Correctness | Detail | Context | Temporal | Consistency |
|-----------|---|---|---|---|---|---|---|---|---|
| [Baseline](https://huggingface.co/sfsdfsafsddsfsdafsa/llama-vid-7b-full-224-long-video-MovieLLM/tree/main/llama-vid-7b-full-224-long-video-baseline) | 49.3 | 3.169 | 43.5 | 2.865 | 1.94 | 2.431 | 2.701 | 1.585 | 1.699
| [Ours](https://huggingface.co/sfsdfsafsddsfsdafsa/llama-vid-7b-full-224-long-video-MovieLLM/tree/main/llama-vid-7b-full-224-long-video-MovieLLM) | 56.7 | 3.46 | 51.3 | 3.141 | 2.154 | 2.549 | 2.88 | 1.832 | 1.976

For MSVD-QA evaluation:
```bash
bash scripts/video/eval/msvd_eval.sh
```
For MSRVTT-QA evaluation:
```bash
bash scripts/video/eval/msrvtt_eval.sh
```

### Long video
To run long video evaluation, please first download corresponding [test-data](https://huggingface.co/datasets/sfsdfsafsddsfsdafsa/Long-video-test-data/tree/main) and [QAs](https://huggingface.co/datasets/sfsdfsafsddsfsdafsa/Long-video-test-data/tree/main).

Then run the following to generate answers for two models (our evaluation methods compare two answers based on reference answer)
```bash
python llamavid/serve/run_llamavid_movie_answer.py --model-path <your-model-path> --video-file <test-data-path> --output_path <path-for-saving-answers> --load-4bit --meta_path <QA-path>
```
Note that in this paper, we run the above for both baseline model and models trained on our data. So, basically, you should have two folders for answers of both models.

Now, you should have following three folders for ground truth, prediction from model 1, prediction from model 2 like the following:

```
res
|-- baseline
|-- ground_truth
|-- ours
```

Then run

```bash
 python eval_movie_qa.py --output_dir ./test/compare_res --api_key <your-api-key> --gt_dir ./res/ground_truth --method_dir ./res/ours --base_dir ./res/basline
```

Finally

```bash
python calculate.py --path ./test/compare_res 
```

#### Results for long video
<image src="docs/long_video_res.png" />




## Results
### Generation Results
<image src="docs/res1.png" />

### Comparison Results
<image src="docs/res2.png" />



## Citation
If you find our work useful, please consider citing:

```bib
@misc{song2024moviellm,
      title={MovieLLM: Enhancing Long Video Understanding with AI-Generated Movies}, 
      author={Zhende Song and Chenchen Wang and Jiamu Sheng and Chi Zhang and Gang Yu and Jiayuan Fan and Tao Chen},
      year={2024},
      eprint={2403.01422},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgement
We would like to thank the following repos for their great work:

- Our experiment is conducted based on [LLaMA-VID](https://github.com/dvlab-research/LLaMA-VID/).
- We perform short video evaluation based on [Video-ChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT).
- We build our pipeline based on [textual-inversion](https://github.com/oss-roettger/XL-Textual-Inversion)

