conda create -n signlang-hf python=3.8.1 -y
conda activate signlang-hf
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install jsonargparse
pip install transformers
pip install evaluate

