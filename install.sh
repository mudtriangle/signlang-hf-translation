conda create -n signlang-hf python=3.8.1 -y
conda activate signlang-hf
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install jsonargparse
pip install transformers==4.30.2
pip install accelerate -U
pip install evaluate
pip install jiwer

