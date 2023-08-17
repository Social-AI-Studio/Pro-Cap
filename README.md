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
@inproceedings{ cao2023procap,
    title={Pro-Cap: Leveraging a Frozen Vision-Language Model for Hateful Meme Detection},
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
The code is tested with python 3.8. To run the code, you should install the package of transformers provided by Huggingface. The code is implemented with the CUDA of 11.2 (you can also implement with other compatible versions) and takes one Tesla V 100 GPU card (with 32G dedicated memory) for model training and inference.

###
## Prepare Datasets  
We have tested on three benchmarks for hateful meme detection: *Facebook Hateful Meme* (FHM), *Multimedia Automatic Misogyny Identification* (MAMI) and *Harmful Memes* (HarM). Datasets are available online. You can either download datasets via links in the original dataset papers or use the files in the **Data** folder provided by us.

For memes, we conduct data pre-processing such as image resizing, text detection and removal and image impainting according to the [HimariO's project][ronzhu]. In our augmentation setting (i.e., augmentation of entities and demographic, see Section 5.3 for details), we detect entities with Google Vision API and conduct face recognition with FairFace. All augmented information is included in our provided data in the **Data** folder

## Pro-Cap Generation

### Step 1: Preprocessing of Datasets

### Step 1: Prompt Frozen PT-VLMs


## Experiments
<p align="center">
  <img src="final-results.JPG" width="80%" height="80%">
</p>
### Performance of Models

### BERT with Pro-Cap

[paper]: https://arxiv.org/abs/2308.08088
[ronzhu]: https://github.com/HimariO/HatefulMemesChallenge
