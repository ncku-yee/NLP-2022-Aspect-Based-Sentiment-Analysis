# NLP-2022-Aspect-Based-Sentiment-Analysis
For NTU NLP 2022 Final Project

---

## Task Description
There are 2 task: **Aspect Category Detection** and **Aspect Category Sentiment Classification**  
1. Aspect Category Detection: Given an input text and a predefined set of aspect categories, identify the aspect categories discussed in the text.  
2. Aspect Category Sentiment Classification: Given an input text and the aspect
categories presented in the text, determine the polarity (positive, negative and neutral) of each aspect category discussed in the text.  

---

## Environment
<table>
<tr>
    <td>Operating System</td>
    <td>Ubuntu 20.04.03 LTS (Focal Fossa)</td>
</tr>
<tr>
    <td>GPU</td>
    <td>Nvidia RTX 3090</td>
</tr>
<tr>
    <td>Nvidia Driver Version</td>
    <td>470.74</td>
</tr>
<tr>
    <td>Python Version</td>
    <td>3.8.10</td>
</tr>
<tr>
    <td>CUDA Version</td>
    <td>11.1</td>
</tr>
</table>

## Dataset
3 dataset: `train.csv`, `dev.csv` and `test.csv` in `data` folder.

---

## Environment Setup
Run below commad:
```bash
pip3 install -r requirements.txt
```
or check if you have below packages in your environment.
```
accelerate
numpy
torch
transformers
pandas
PyYAML
scikit-learn
tqdm
```

---

## Model Architecture
Model architecture is in [Network.py](https://github.com/ncku-yee/NLP-2022-Aspect-Based-Sentiment-Analysis/blob/master/Network.py).  
By default, class `MultilabelClassifier` is for **task1** and class `SentimentClassifier` is for **task2**.
If you want to add the pretrained model name provided on [huggingface](https://huggingface.co/models), please add the MODEL NAME to `MODELS` in this python script.

---

## Configuration
Configuration files for each task are in `configs/task<TASK_IDENTIFIER>` folder.  
The default configuration is in [Config.py](https://github.com/ncku-yee/NLP-2022-Aspect-Based-Sentiment-Analysis/blob/master/Config.py)  
You can add any configuration in `YAML` format.  
If you want to run with your customized configuration, run the below command:  
```bash
python3 train.py --config <PATH-TO-YOUR-CONFIG>
```

---

## Ensemble
Revise the pretrained model list in [ensemble.py](https://github.com/ncku-yee/NLP-2022-Aspect-Based-Sentiment-Analysis/blob/master/ensemble.py)  
The revise the `TASK_IDENTIFIER` in `ensemble_script.sh` and other configurations.  
Final, just run below command:  
```bash
./ensemble_script.sh
```
---