# Procedural Reasoning Networks for Understanding Multimodal Procedures
<p align="center">
  <b>Mustafa Sercan Amac, Semih Yagcioglu, Aykut Erdem, Erkut Erdem</b></span>
</p>

This is the implementation of [ Procedural Reasoning Networks for Understanding Multimodal Procedures](https://www.aclweb.org/anthology/K19-1041/) (CoNLL 2019) on a RecipeQA dataset: [RecipeQA dataset]([https://hucvl.github.io/recipeqa/](https://hucvl.github.io/recipeqa/)) . We propose Procedural Reasoning Networks (PRN) to address the problem of comprehending procedural commonsense knowledge.
See our [website]([https://hucvl.github.io/prn/](https://hucvl.github.io/prn/))  for more information about the model!

<div align="center">
  <img src="https://hucvl.github.io/prn/index_files/overview_v5.png?raw=true" style="float:left" width="100%">
</div>

## Bibtex
For PRN:
```
@inproceedings{prn2019,
  title={Procedural Reasoning Networks for Understanding Multimodal Procedures},
  author={Amac, Mustafa Sercan and Yagcioglu, Semih and Erdem, Aykut and Erdem, Erkut},
  booktitle={Proceedings of the CoNLL 2019},
  year={2019}
}
```

For RecipeQA dataset:
```
@inproceedings{yagcioglu-etal-2018-recipeqa,
   title = “{R}ecipe{QA}: A Challenge Dataset for Multimodal Comprehension of Cooking Recipes”,
   author = “Yagcioglu, Semih  and
     Erdem, Aykut  and
     Erdem, Erkut  and
     Ikizler-Cinbis, Nazli”,
   booktitle = “Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing”,
   year = “2018",
   address = “Brussels, Belgium”,
   publisher = “Association for Computational Linguistics”,
   url = “https://www.aclweb.org/anthology/D18-1166“,
   doi = “10.18653/v1/D18-1166”,
   pages = “1358--1368",
}
```

## Requirements
- You would need python3.6 or python3.7
- See [`requirements.txt`](requirements.txt) for the required python packages and run `pip install -r requirements.txt` to install them.

## Pre-processing
The code will automatically download pre-trained features and start the pre-processing procedure.


## Training 
To train the model, run the following command:
```bash
allennlp train config_file -s  directory_to_save --include-package recipeqalib
```
### Example
We prepared 2 example config files. One of them is for single-task training, and the other one is for multi-task training. For training the single-task model run the following command:
```bash
allennlp train ./configs/example_single_task.json -s ./save/example_single_task --include-package recipeqalib
```
For training the multi-task model run the following command:
```bash
allennlp train ./configs/example_multi_task.json -s ./save/example_multi_task --include-package recipeqalib
```
## Evaluation
In order to evaluate the trained model you would need to test image features and the test set questions. You can download them with the following script.
```bash
wget https://vision.cs.hacettepe.edu.tr/files/recipeqa/test.json ./data/test.json
wget https://vision.cs.hacettepe.edu.tr/files/recipeqa/test_img_features.pkl ./data/test_img_features.pkl
```
For a step-by-step evaluation example please see the evaluate_model notebook under notebooks folder.
