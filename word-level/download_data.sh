
# Download glove embeddings

mkdir -p data/pretr/emb 2> /dev/null
cd data/pretr/emb
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
rm glove.6B.zip glove.6B.200d.txt glove.6B.300d.txt glove.6B.50d.txt
cd ../../..