### 0. Environment
First, please create an environment.
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

### 1. Test Our Performance!
This command is especially crucial, because after you run it, you can get the final result of our team. The --input is followed by the test input, and the test output file name is "task2output.xml" by default, which you can also set by --output. If you don't know how to do with this, please contact us!
~~~
# test command for our TA
# Your need to change --input and --output to your directory
python postprocess.py --model bert --ckpt ./ckpt/ccf_bert_cn_best.pth --lang cn --input ./demo/task2input_cn.xml --tuning 1 --output task2output_cn.xml 
python postprocess.py --model bert --ckpt ./ckpt/ccf_bert_en_best.pth --lang en --input ./demo/task2input_en.xml --tuning 1 --output task2output_en.xml
~~~

### 2. Set up for training
Now that you've created a ready to go python environment, you can set up the code base and prepare the data for training.
~~~
cd data 
mkdir CCF # md CCF (for windows)
cd CCF
mkdir json # md json (for windows)
cd ../../
python preprocess.py
~~~

### 3. Train
~~~
# train on ccf training set
python train.py --dataset ccf --model simple --lang en
python train.py --dataset ccf --model bert --lang en
python train.py --dataset ccf --model bert --lang cn
# train with fine tune
python train_tune.py --dataset ccf --model fc --lang cn --tuning 1
python train_tune.py --dataset ccf --model bert --lang cn --tuning 1
python train_tune.py --dataset ccf --model bert --lang en --tuning 1
# fine tune with mixout
python train_tune.py --dataset ccf --model bert --lang cn --tuning 1 --mixout 1

# train on imdb training set (English)
python train.py --dataset imdb --model simple --lang en
python train.py --dataset imdb --model bert --lang en
~~~

### 4. Evaluate on Checkpoint File
~~~
# evaluate on ccf dataset or imdb dataset
# intradomain testing
python evaluate.py --dataset ccf --model simple --ckpt ./ckpt/ccf_simple_en_best.pth --lang en
python evaluate.py --dataset ccf --model bert --ckpt ./ckpt/ccf_bert_en_best.pth --lang cn
python evaluate.py --dataset ccf --model bert --ckpt ./ckpt/ccf_bert_cn_best.pth --lang en
# evaluate on fine tuned models
python evaluate.py --dataset ccf --model bert --ckpt ./ckpt/ccf_bert_cn_best.pth --lang cn --tuning 1
python evaluate.py --dataset ccf --model bert --ckpt ./ckpt/ccf_bert_en_best.pth --lang en --tuning 1

# interdomain testing
python evaluate.py --dataset imdb --model simple --ckpt ./ckpt/ccf_simple_en_best.pth --lang en
python evaluate.py --dataset imdb --model bert --ckpt ./ckpt/ccf_bert_en_best.pth --lang en
~~~

### 5. Show the Results on Train Set
~~~
# Chinese post processing
python postprocess.py --model bert --ckpt ./ckpt/ccf_bert_cn_best.pth --lang cn --input ./data/CCFtrain02/cn_sample_data/sample.negative.txt --tuning 1
python postprocess.py --model bert --ckpt ./ckpt/ccf_bert_cn_best.pth --lang cn --input ./data/CCFtrain02/cn_sample_data/sample.positive.txt --tuning 1

# English post processing
python postprocess.py --model bert --ckpt ./ckpt/ccf_bert_en_best.pth --lang en --input ./data/CCFtrain02/en_sample_data/sample.negative.txt --tuning 1
python postprocess.py --model bert --ckpt ./ckpt/ccf_bert_en_best.pth --lang en --input ./data/CCFtrain02/en_sample_data/sample.positive.txt --tuning 1

~~~

### 6. Play with Demo
~~~
python demo.py --model bert --ckpt ./ckpt/ccf_bert_cn_best.pth --lang cn --tuning 1
~~~