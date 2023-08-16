# Pro-Cap: Leveraging a Frozen Vision-Language Model for Hateful Meme Detection

This includes an original implementation of "[Pro-Cap: Leveraging a Frozen Vision-Language Model for Hateful Meme Detection][paper]" by Rui Cao, Ming Shan Hee, Adriel Kuek, Wen-Haw Chong, Roy Ka-Wei Lee, Jing Jiang.

<p align="center">
  <img src="archmm.png" width="80%" height="80%">
</p>

This code provides:
- Codes for generating Pro-Cap according to probing questions with frozen pre-trained vision-language models (PT-VLMs).
- Commands to run the models and get numbers reported in main experiment of the paper.

Please leave issues for any questions about the paper or the code.

If you find our code or paper useful, please cite the paper:
```
@inproceedings{ lyu2022zicl,
    title={ Z-ICL: Zero-Shot In-Context Learning with Pseudo-Demonstrations },
    author={ Rui Cao, Ming Shan Hee, Adriel Kuek, Wen-Haw Chong, Roy Ka-Wei Lee, Jing Jiang},
    journal={ ACM MM },
    year={ 2023 }
}
```

### Announcements
07/26/2023: Our paper is accepted by ACM MM 2023. 

## Content
1. [Installation](#installation)
2. [Prepare Datasets](#prepare-datasets)
3. [Pro-Cap Generation](#pro-cap-generation) (Section 4.2 of the paper)
    * [Step 1: Pre-processing of Datasets](#step-1-preprocessing-of-datasets) (Section 4.2 of the paper)
    * [Step 2: Prompt Frozen PT-VLMs](#step-2-prompt-frozen-ptvlms) (Section 5.2 of the paper)
4. [Experiments](#experiments) (Section 4 of the paper)
    * [PromptHate with Pro-Cap](#prompthate-with-procap)
    * [BERT with Pro-Cap](#bert-with-procap)

## Installation
The code is tested with python 3.8.

Create a new conda environment for this project:
```bash
conda create --name z-icl python=3.8
conda activate z-icl
```

Install the dependencies (`gdown`, `datasets`, `faiss`, `nltk`, `tqdm`, `simcse`, `pytorch`, `transformers`) using the following command:
```bash
bash build_dependencies.sh
```

###
## Prepare Datasets  

## Pro-Cap Generation

### Step 1: Preprocessing of Datasets

### Step 1: Prompt Frozen PT-VLMs


## Experiments

### PromptHate with Pro-Cap

### BERT with Pro-Cap
[paper]: https://arxiv.org/abs/2212.09865
