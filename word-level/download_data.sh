
# Download conll03 dataset
mkdir -p data/dset/03co 2> /dev/null
cd data/dset/03co
curl -O https://raw.githubusercontent.com/Franck-Dernoncourt/NeuroNER/master/neuroner/data/conll2003/en/train.txt
curl -O https://raw.githubusercontent.com/Franck-Dernoncourt/NeuroNER/master/neuroner/data/conll2003/en/valid.txt
curl -O https://raw.githubusercontent.com/Franck-Dernoncourt/NeuroNER/master/neuroner/data/conll2003/en/test.txt
cd ../../..
python preprocess.py

# Download glove embeddings
mkdir -p data/pretr/emb 2> /dev/null
cd data/pretr/emb
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
rm glove.6B.zip glove.6B.200d.txt glove.6B.300d.txt glove.6B.50d.txt
cd ../../..
