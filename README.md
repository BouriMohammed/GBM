# Bridging Robustness and Generalization Against Word Substitution Attacks in NLP via the Growth Bound Matrix Approach

This repository contains the official implementation of the paper  
**_"Bridging Robustness and Generalization Against Word Substitution Attacks in NLP via the Growth Bound Matrix Approach"_**  
by *Mohammed Bouri* and *Adnane Saoud*, accepted at **Findings of Association for Computational Linguistics (ACL)**.

## 📚 Table of Contents

- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [GBM Extraction from Pre-Trained Models](#-gbm-extraction-from-pre-trained-models)
- [Model Training with GBM](#-model-training-with-gbm)
  - [Dataset](#-dataset)
  - [Word Embeddings](#-word-embeddings)
  - [Data Preprocessing](#-data-preprocessing)
- [Training](#-training)
  - [BiLSTM + GBM](#-bilstm--gbm)
  - [CNN + GBM](#-cnn--gbm)
- [Robustness Evaluation](#-robustness-evaluation)
  - [BiLSTM Attack (PWWS)](#-bilstm-attack-pwws)
  - [CNN Attack (PWWS)](#-cnn-attack-pwws)
- [S4 Model + GBM](#-s4-model--gbm)
  - [Installation](#-installation)
  - [Training](#-training-1)
  - [Robustness (PWWS)](#-robustness-pwws)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

## 📘 Overview

This project introduces the **Growth Bound Matrix (GBM)** approach, a theoretically grounded method to enhance both **robustness** and **generalization** of NLP models against **word substitution attacks**.

We provide:
- Tools for extracting GBMs from pre-trained models
- GBM-based training routines for BiLSTM, CNN, and S4 architectures
- Robustness evaluations using **OpenAttack** and **TextAttack**



## 📁 Project Structure

```
GBM/
├── data/                    # IMDB data and embeddings
├── inputs/                  # Inputs's Interval
├── model/
├── OpenAttack/              # Attack framework
├── TextAttack/              # For S4 compatibility
├── s4/                      # S4 model implementation
├── script_numba/            # Acceleration utilities
├── attacks_BILSTM.py        # Attacks on BiLSTM
├── attacks_CNN.py           # Attacks on CNN
├── attacks_S4.py            # Attacks on S4
├── bert_attack.py           # BERT-specific attack
├── data.py                  # Data processing
├── GBM_BILSTM.py            # BiLSTM + GBM training
├── GBM_CNN.py               # CNN + GBM training
├── GBM_S4.py                # S4 + GBM training
├── GBM_test.ipynb           # GBM extraction demo
├── requirements.txt
├── readme.md
```



## 📊 GBM Extraction from Pre-Trained Models

Run the Jupyter notebook [`GBM_test.ipynb`](GBM_test.ipynb) for a step-by-step tutorial on how to extract GBM matrices from any pre-trained model.



## 🧠 Model Training with GBM

### 🧾 Dataset

- Download the IMDB dataset and place it under `./data/imdb/`.  
  [📂 IMDB Dataset (Google Drive)](https://drive.google.com/drive/folders/13ZzX4uP1gUyeLbZBALV0aXmD2aLqgE07?usp=sharing)

### 🔤 Word Embeddings

- Download 300-dimensional **GloVe embeddings** and place them under `./data/embedding/`.  
  [🔗 GloVe Embeddings](http://nlp.stanford.edu/data/glove.840B.300d.zip)

### ⚙️ Data Preprocessing

```bash
python data.py
```



## 🏋️ Training

### BiLSTM + GBM

```bash
python GBM_BILSTM.py -T 50 -ka 0.6 -kb 0.3 >> logs_gbm_bilstm.txt
```

### CNN + GBM

```bash
python GBM_CNN.py -T 50 -k 0.9 >> logs_gbm_cnn.txt
```


## 🛡️ Robustness Evaluation

### BiLSTM Attack (PWWS)

```bash
python attacks_BILSTM.py --attack pwws -ka 0.6 -kb 0.3 >> logs_attack_pwws_gbm_bilstm.txt
```

### CNN Attack (PWWS)

```bash
python attacks_CNN.py --attack pwws -k 0.9 >> logs_attack_pwws_gbm_cnn.txt
```

> **Note:** Attacks on BiLSTM and CNN models are conducted using the [OpenAttack](https://github.com/thunlp/OpenAttack) framework.


## 📈 S4 Model + GBM

### Installation

Install S4 from the [official repository](https://github.com/state-spaces/s4) before proceeding.

### Training

```bash
python GBM_S4.py -T 200 --learning_rate 0.0005 --patience 7 -d 256 --num_layers 1 --checkpoint -ka 0.1 -kb 0.2 > logs_gbm_s4.txt
```

### Robustness (PWWS)

```bash
python3 attacks_S4.py --sample 500 -d 256 --attack pwws -ka 0.2 -kb 0.1 > logs_attack_pwws_gbm_s4.txt
```

> **Note:** Attacks on S4 are conducted using [TextAttack](https://github.com/QData/TextAttack) to ensure compatibility with the S4 implementation.

## 📖 Citation

If you use this code or find our work helpful in your research, please cite our paper:

```bibtex
@inproceedings{bouri-saoud-2025-bridging,
    title = "Bridging Robustness and Generalization Against Word Substitution Attacks in {NLP} via the Growth Bound Matrix Approach",
    author = "Bouri, Mohammed  and
      Saoud, Adnane",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.627/",
    doi = "10.18653/v1/2025.findings-acl.627",
    pages = "12118--12137",
    ISBN = "979-8-89176-256-5"```
```

## 🤝 Contributions

We welcome contributions! If you'd like to help improve this project, feel free to open issues for bugs, questions, or enhancement ideas.

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

## 📬 Contact

If you have any questions, suggestions, or feedback, feel free to contact us at: [mohammed.bouri@um6p.ma](mailto:mohammed.bouri@um6p.ma)

## 🤝 Acknowledgements

- [OpenAttack](https://github.com/thunlp/OpenAttack)
- [TextAttack](https://github.com/QData/TextAttack)
- [S4: State Space Models for Sequences](https://github.com/state-spaces/s4)
- [GloVe Embeddings](https://nlp.stanford.edu/projects/glove/)
