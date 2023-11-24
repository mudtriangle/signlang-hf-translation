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

## Performance
The following is a table detailing all the variations submitted to the challenge's leaderboard. Our final winning submission is in **bold**.
| Backbone | Sampling Strategy | BLEU | chrF |
| :--- | :---: | :---: | :---: |
| T5-efficient-large | Greedy | 0.9 | 16.0 |
| T5-efficient-large | Top-$k$ | 0.8 | 16.3 |
| T5-efficient-large | Beam | 0.9 | 17.2 |
| T5-efficient-large | Top-$k$ Beam | 0.8 | 17.3 |
| **T5-efficient-large** | **Diverse Beam** | **1.1** | **17.0** |

## Representations and Checkpoints
Coming soon!

## Citation
If you find this repository helpful, consider citing our system paper:
```
[Citation pending]
```

