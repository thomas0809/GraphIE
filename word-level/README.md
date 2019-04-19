# GraphIE: This is the code for word-level GCN

This code is implemented by [Zhijing Jin](http://zhijing-jin.com/fantasy/about/). For any questions, please contact via [email](mailto:zhijing.jin@connect.hku.hk).

The original framework is accreditted to [Di Jin](https://github.com/jind11).

## Get Started

### 1) Install packages
- Python>=3.6
- [PyTorch 0.4.0](https://pytorch.org/) # install it according to your cuda version. 
e.g. `conda install pytorch=0.4.0 torchvision cuda80 -c pytorch`
```bash
conda create -n graphie_env python=3.6
conda activate graphie_env
pip install -r requirements.txt
# install pytorch 0.4.0 in your own way
```

### 2) Download data
```bash
./download_data.sh # (1) download and preprocess Conll2003; (2) download Glove embeddings
```
Note: If you need to preprocess new data sources, please see
- Small [data samples](data/dset/sample_format) are provided.
- If have any questions regarding `preprocess.py`, you can contact the author by [email](mailto:zhijing.jin@connect.hku.hk).

## Run
```
python examples/multi_runs_conll.py --gpu_id 0
```


## Model Outputs
### Overview
To ensure the files are correct, I did the following checks on `Apr 2, 2019`.

|Conll03 Dev/Test|# docs|GraphIE (best)|SeqIE (best)|
|---|---|---|---|
|Test|231|92.02|91.57|
|Dev|216|94.90|94.66|

### Output Files
|Conll03 Dev/Test|GraphIE (best)|SeqIE (best)|
|---|---|---|
|Test|[Output](outputs/conll03_GraphIE/11131230r10_test) (92.02)|[Output](outputs/conll03_SeqIE/11141152r09_test)(91.57)|
|Dev|[Output](outputs/conll03_GraphIE/11131230r10_dev) (94.90)|[Output](outputs/conll03_SeqIE/11141152r09_dev)(94.66)|

### Specs
- You can evaluate the outputs by running the official perl script to calculate NER F1 scores:
```
code/examples/eval/conll03eval.v2 < outputs/conll03_SeqIE/11141152r09_dev
```
- In each output file, the format is `index \t word \t gold_tag \t predicted_tag`. The start of a document is signified by `1 -DOCSTART- O O`.
- The dev output is saved in the last epoch in the training.
- The test output is saved in the epoch with the highest F1 score on the dev data.
- To get a better understanding of how the outputs are saved, please refer to the main [code file](examples/NERCRF_conll.py).

## Appendix: Full experiment details
### conll03 - GraphIE
To show a robust result, we run the code multiple times and gathered the following outputs:

|filename|Test F1|
|---|---|
|11131129r09 | 91.77 |
|11131158r09 | 91.96 |
|11131159r09 | 91.34 |
|11131229r10 | 91.63 |
|11131230r10 | 92.02 |
|11141004r03 | 92 |

### conll03 - SeqIE
To show a robust result, we run the code multiple times and gathered the following outputs:

|filename|Test F1|
|---|---|
|11141150r09 | 91.45 |
|11141152r09 | 91.57 |
|11141153r09 | 90.72 |
|11141154r11 | 90.94 |
|11141156r11 | 91.14 |



