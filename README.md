# GraphIE

## Requirements

* python
* PyTorch
* tqdm, pandas, dateutil

## Preprocessing

* Parse the pdf with PDFMiner
  ```
  python parse_pdf.py
  ```
  By default, all the pdfs are stored in `Content/`, while each case in a separate folder. Please edit the code if the pdfs
  are stored elsewhere.

* Generate the graph and annotate the tags
  ```
  python gen_graph.py --base Content/ --excel samples_labels.xlsx
  ```

## Training
  ```
  python train.py --base ./ --case selected_case.txt
  ```
  Please see the code for the parameter definitions.

## Testing
  ```
  python train.py --test  --testmodel xxx.model --testbase ./ --testcase test.txt
  ```
  
## Postprocessing
  Generate the predictions to a .xlsx file and evaluate.
  ```
  python evaluate.py --base ./ --case test.txt --model gnn
  ```
  By default, it performs automatic evaluation using the ground truth in `Cons_Full_training.xlsx`.
  If evaluation is not needed, please edit the code.
