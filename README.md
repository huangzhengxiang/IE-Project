### 0. Environment
~~~
# create environment
conda create -n nlp python==3.7.2
conda activate nlp
# install pytorch
conda install pytorch==1.10.2 torchvision==0.11.3 torchaudio==0.10.2 cudatoolkit=11.3 -c pytorch -c pytorch-lts
pip install pandas
pip install lxml
pip install torchtext==0.11.2
pip install typing-extensions==4.1.1
# update wheel
e.g {C:\Users\Dell\.conda\envs\nlp\python.exe} -m pip install -U pip setuptools wheel
# install other dependencies (spacy==3.4.2)
pip install -U spacy==3.4.2
python -m spacy download zh_core_web_sm
python -m spacy download en_core_web_sm
pip install transformers==4.21.0
pip install pillow==9.3.0
~~~
### 1. Set up
First, download GloVe pre-trained word-embedding from , unzip it in the current directory.
Then, build vocabulary.
~~~
cd data 
mkdir CCF # md CCF (windows)
cd CCF
mkdir json # md json (windows)
cd ../../
python build_data.py --pkl 0 --json 1
~~~
~~~
<!-- python build_data.py --pkl 0 --json 1 -->
python preprocess.py
~~~

~~~
# train on ccf training set
python train.py --dataset ccf --model simple --lang en
python train.py --dataset ccf --model bert --lang en
python train.py --dataset ccf --model bert --lang cn
# train on imdb training set
python train.py --dataset imdb --model simple --lang en
python train.py --dataset imdb --model bert --lang en
~~~

~~~
# evaluate on ccf dataset or imdb dataset
# intradomain testing
python evaluate.py --dataset ccf --model simple --ckpt ./ckpt/ccf_simple_en_best.pth --lang en
python evaluate.py --dataset ccf --model bert --ckpt ./ckpt/ccf_bert_en_best.pth --lang en
python evaluate.py --dataset ccf --model bert --ckpt ./ckpt/ccf_bert_cn_best.pth --lang cn
# interdomain testing
python evaluate.py --dataset imdb --model simple --ckpt ./ckpt/ccf_simple_en_best.pth --lang en
python evaluate.py --dataset imdb --model bert --ckpt ./ckpt/ccf_bert_en_best.pth --lang en
~~~
