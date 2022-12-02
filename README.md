### 0. Environment
~~~
# create environment
conda create -n nlp python==3.7.2
conda activate nlp
# install pytorch
conda install pytorch==1.10.2 torchvision==0.11.3 torchaudio==0.10.2 cudatoolkit=11.3 -c pytorch -c pytorch-lts
pip install pandas
pip install torchtext==0.11.2
pip install typing-extensions==4.1.1
# update wheel
e.g {C:\Users\Dell\.conda\envs\nlp\python.exe} -m pip install -U pip setuptools wheel
# install other dependencies
pip install -U spacy
python -m spacy download zh_core_web_sm
python -m spacy download en_core_web_sm
pip install transformers==4.21.0
pip install pillow==9.3.0
~~~
### 1. Set up
First, download GloVe pre-trained word-embedding from , unzip it in the current directory.
Then, build vocabulary.
~~~
python build_data.py --pkl 0 --json 1
~~~

~~~
# train on ccf training set
python train.py --dataset ccf --model simple
python train.py --dataset ccf --model bert
# train on imdb training set
python train.py --dataset imdb --model simple
python train.py --dataset imdb --model bert
~~~

~~~
# evaluate on ccf dataset or imdb dataset
# intradomain testing
python evaluate.py --dataset ccf --model simple --ckpt ./ckpt/ccf_simple_best.pth
python evaluate.py --dataset ccf --model bert --ckpt ./ckpt/ccf_bert_best.pth
# interdomain testing
python evaluate.py --dataset imdb --model simple --ckpt ./ckpt/ccf_simple_best.pth
python evaluate.py --dataset imdb --model bert --ckpt ./ckpt/ccf_bert_best.pth 
~~~