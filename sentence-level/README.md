# GraphIE (sentence-level)

## Requirements

* python 3.6
* PyTorch 0.4.1
* torchtext
* tqdm, termcolor

## Social Media Information Extraction

The datasets (Education, Job) are constructed from the [corpus](https://nlp.stanford.edu/~bdlijiwei/Code.html) used in [Weakly Supervised User Proﬁle Extraction from Twitter (ACL'14)](http://aritter.github.io/acl2014_li.pdf). See the `twitter_data/` folder.

To reproduce the experiments, run
```
python dispatcher_twitter.py --num_gpu=2 --task education
```

## Visual Information Extraction

We cannot release the data for patient privacy and proprietary reasons. The codes can be however found in the `scripts-for-visual-ie` folder. 

`parse_pdf.py` is for parsing PDFs using [pdfminer](https://github.com/euske/pdfminer). 

`gen_graph_bio.py` constructs the graph and converts the answers to tags. 

`gen_vocab.py` generates the vocabulary. 

`train.py` is the core code for training and testing. (Note that `gnn.py`, `ModelUtils.py`, `attention.py` are also required.)
