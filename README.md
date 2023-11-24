# TTIC's Submission to WMT-SLT23
## Overview
Highest rated model by all automatic metrics and human evaluation!

This repository is based on the [HuggingFace](https://huggingface.co) ecosystem, and uses representations learned with [BEVT](https://github.com/xyzforever/BEVT) video-only self-supervised training. Our main translation architecture is an adaptation of the [T5](https://arxiv.org/abs/1910.10683) architecture, using the pre-trained checkpoints provided by the [GermanT5 group](https://huggingface.co/GermanT5).

For a detailed description of the overall submission, please see the system paper.

## Data
To download the data used in the challenge, kindly follow the instructions provided by the challenge organisers [on their website](https://www.wmt-slt.com/data). We preprocess this data by removing subtitles that include the string `WEBVTT` as part of their body, which we found to be detrimental to translation quality.

## Setup
To set up a conda environment with the necessary libraries to run this code, please execute the following command:
```
sh install.sh
```

## Running the Code
Before running the code, please modify the files inside the directory `run-configs` with the corresponding paths for `run_dir`, `model_config`, `model_checkpoint`, `train_files`, `train_labels`, `valid_files`, `valid_labels`, and `features_dir`. The expected list of files is a TSV file with the following format per line: `[video_name]\t[num_of_frames]`. The corresponding label files are text files with one translation per line, where the sentence in the first line corresponds to the file in the first line of the list of files.

To extract the features from BEVT representations, please follow their repository [here](https://github.com/xyzforever/BEVT) using one of our pre-trained BEVT checkpoints. These must all be extracted directly in the folder specified by `features_dir`.

Lastly, to execute our training code, you can simply use `torchrun`. For example, to run in 8 GPUs (our default setup), you can do so with the following commands:
```
torchrun --nproc_per_node 8 train.py --config run-configs/t5/bevt-oasl-efficient-large-stage1.yaml
torchrun --nproc_per_node 8 train.py --config run-configs/t5/bevt-oasl-efficient-large-stage2.yaml
```

## Performance
The following is a table detailing all the variations submitted to the challenge's leaderboard. Our final winning submission is in **bold**.
| Backbone | Sampling Strategy | BLEU | chrF |
| :--- | :---: | :---: | :---: |
| T5-efficient-large | Greedy | 0.9 | 16.0 |
| T5-efficient-large | Top-k | 0.8 | 16.3 |
| T5-efficient-large | Beam | 0.9 | 17.2 |
| T5-efficient-large | Top-k Beam | 0.8 | 17.3 |
| **T5-efficient-large** | **Diverse Beam** | **1.1** | **17.0** |

## Representations and Checkpoints
Coming soon!

## Citation
If you find this repository helpful, consider citing our system paper:
```
[Citation pending]
```
