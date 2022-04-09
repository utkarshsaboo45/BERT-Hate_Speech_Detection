# Hate Speech Detection in Online Social Media using Transfer Learning

## Progress Report I

### Introduction

#### Aim

Our aim is to train and compare multiple models for hate-speech-detection multi-class-classification problem. We train these models on raw (social media) text to classify it among classes such as `hate_speech`, `offensive_language` and `neither`.

#### Tasks

- We use several pre-trained models, specifically `FastText`, `BERT`, `BERT-CNN` and `BERTweet` and adapt them to our problem of hate-speech classification
- As a part of our baseline experiments, we use `FastText` and compare it with different flavours of BERT to see which one performs the best for this particular domain of NLP.
- Finally, we use the above mentioned models to classify text from an entirely separate dataset, one which still contains social-media text, but with text which is different from what our model has seen during training, and check how well these models generalize over different variations of social-media text.

---

### Motivation and Contributions

#### Motivation

The interaction among users on different social media platforms generate a vast amount of data. Users of these platforms often indulge in detrimental, offensive and hateful behavior toward numerous segments of the society. While hate speech is not a recent phenomenon, the recent surge in online forums allows perpetrators to more directly target their victims. In addition, hate speech may polarize public opinion and hurt political discourse, with detrimental consequences for democracies. Therefore, the primary motivation for this project is an effort to build an automated mechanism to detect and filter such speech and create a safer, more user-friendly environment for social media users.

In an attempt to do this, we use multiple pre-trained models and train them to classify text from any social media platform. 

#### Contribution

- Detecting hate speech is a complicated task from the semantics point of view. Moreover, when it comes to middle- and low-resource domains, the research in hate speech detection is almost insignificant due to the lack of labeled data. This has resulted in the emergence of bias in technology.
- Further, the models trained on text from one social media platform, such as twitter, tend not to work too well on texts from other platforms such as Facebook and YouTube.
- To address these issues, we apply transfer learning approach for hate speech detection and train generalized models which can work on cross-platform texts.

---

### Data

We use two datasets for our tasks:
1. [t-davidson_hate-speech-and-offensive-language](https://github.com/t-davidson/hate-speech-and-offensive-language/tree/master/data)
2. [ucberkeley-dlab_measuring-hate-speech](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech)

These two datasets are readily available:
- The first one is available on github. It's in raw for so it needs pre-processing.
- The second one is available publicly on huggingface and can be acquired using the `datasets` library. It's slightly processed but still needs more pre-processing.

Have a peek at the data description [here](https://github.ubc.ca/sneha910/COLX_585_BERT-Fine-Tuning-Hate-Speech-Detection/blob/master/notebooks/data_description.ipynb).

An Exploratory Data Analysis of the datasets can be found [here](https://github.ubc.ca/sneha910/COLX_585_BERT-Fine-Tuning-Hate-Speech-Detection/blob/master/notebooks/EDA.ipynb)

__Note__: To replicate the results in [data_description.ipynb](https://github.ubc.ca/sneha910/COLX_585_BERT-Fine-Tuning-Hate-Speech-Detection/blob/master/notebooks/data_description.ipynb):
- You need to download [this dataset](https://github.com/t-davidson/hate-speech-and-offensive-language/tree/master/data), and place the file `labeled_data.csv` in `data/github` folder.
- You need to install the [datasets](https://pypi.org/project/datasets/) package, as this is the source of one of our datasets.

---

### Engineering

1. ``Computing infrastructure``

Our computing infrastructure includes our personal computers and Google Colab.

2. ``DL-NLP methods``

We use transfer learning by fine-tuning pre-trained models like vanilla BERT, BERT-CNN and BERTweet on our dataset for hate speech classification.
To compare how better these models perform, we set `FastText` as our baseline.

3. ``Framework``

We use to use `PyTorch` as our primary framework. Our models include pre-trained `FastText` and different variations of `BERT` from the `HuggingFace` library.

---

### Previous Works

As the research grows in the field of Hate-speech detection on social media platforms (e.g., in SemEval-2019, one of the major tasks was classifying Twitter data as either hateful or not hateful), many researchers have increasingly shifted focus toward applying Deep Learning models for this task. As a basis for our project, we referred to the following two papers:

[A BERT-Based Transfer Learning Approach for Hate Speech Detection in Online Social Media](https://arxiv.org/pdf/1910.12574.pdf)

This paper talks about a transfer learning approach using the pre-trained language model BERT learned on General English Corpus (no specific domain) to enhance hate speech detection on publicly available online social media datasets. They also introduce new fine-tuning strategies to examine the effect of different embedding layers of BERT in hate speech detection.

[Hate speech detection on Twitter using transfer learning](https://www.sciencedirect.com/science/article/abs/pii/S0885230822000110)

This paper shows that multi-lingual models such as `XLM-RoBERTa` and `Distil BERT ` are largely able to learn the contextual information in tweets and accurately classify hate and offensive speech.

---

### Evaluation

We use F1-scrore as the major metric to evaluate our model. We compare the F1-scores of different models on cross-platform unseen data. The one which gives the best score is the best-suited for classification of generalized social-media text

---

### Conclusion

Our goal materializes from the fact that social media, being a widely used mode to socialize, has become unsafe for people looking for a secure environment to communicate. We come up with an efficient Deep Learning model to detect hate speech in online social media domain data using by fine tuning different variations of BERT pretrained model. This will become a useful tool to filter out any offensive and detrimental content across the social media platforms, even the ones which our model has never seen, and safeguard people from usage of hate speech.

---