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

We use transfer learning by fine-tuning pre-trained models like vanilla BERT, BERT-CNN, RoBERTa and BERTweet on our dataset for hate speech classification.
To compare how better these models perform, we set `FastText` as our baseline.

3. ``Framework``

We use to use `PyTorch` as our primary framework. Our models include pre-trained `FastText` and different variations of `BERT` from the `HuggingFace` library.

---

### Previous Works

As the research grows in the field of Hate-speech detection on social media platforms (e.g., in SemEval-2019, one of the major tasks was classifying Twitter data as either hateful or not hateful), many researchers have increasingly shifted focus toward applying Deep Learning models for this task. As a basis for our project, we referred to the following papers:

[Constructing interval variables via faceted Rasch measurement and multitask deep learning: a hate speech application](https://arxiv.org/abs/2009.10277)

This paper descibes the dataset we decided to use in details. The paper also shows the methods they used to train on that data. We decided to make this a classification problem, but the authors of the paper wanted to put hatespeech on an intensity scale and made it a regression problem. The paper. They also added intermediate outputs to their architecture that they used to predict the final results.

[A BERT-Based Transfer Learning Approach for Hate Speech Detection in Online Social Media](https://arxiv.org/pdf/1910.12574.pdf)

This paper talks about a transfer learning approach using the pre-trained language model BERT learned on General English Corpus (no specific domain) to enhance hate speech detection on publicly available online social media datasets. They also introduce new fine-tuning strategies to examine the effect of different embedding layers of BERT in hate speech detection.

[Hate speech detection on Twitter using transfer learning](https://www.sciencedirect.com/science/article/abs/pii/S0885230822000110)

This paper shows that multi-lingual models such as `XLM-RoBERTa` and `Distil BERT ` are largely able to learn the contextual information in tweets and accurately classify hate and offensive speech.


[BERTweet: A pre-trained language model for English Tweets](https://aclanthology.org/2020.emnlp-demos.2.pdf)

BERTweet is the first public largescale pre-trained language model for English Tweets. This paper shows that BERTweet outperforms strong baselines RoBERTabase and XLM-Rbase, producing better performance results than the previous state-of-the-art models on three Tweet NLP tasks: Part-of-speech tagging, Named-entity recognition and text classification. The model uses the BERTbase model configuration, trained based on the RoBERTa pre-training procedure. The authors used an 80GB pre-training dataset of uncompressed texts, containing 850M Tweets (16B word tokens), where each Tweet consists of at least 10 and at most 64 word tokens.

---

### Results

In this section, we present discussion on our results obtained from different models. For the purpose of acquiring some baseline benchmark results on the dataset, we have used following models: 

#### FastText (baseline)

This is the baseline we decided to use to compare other models. The reason we chose the FastText classifier is because it's a simple fast to train linear model.

The data used to train the model (without hatespeech = 0):

`Train data size: 108444`
`Test data size: 27112`

With 50 epochs and a learning rate of 0.01 gave:

`Precision Score: 0.8321579133997873`
`F1 Score: 0.7675433809887591`

#### BERTweet 

We are using [ucberkeley-dlab_measuring-hate-speech](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech) as our dataset. Our dataset was normalized (translating emotion icons into text strings, converting user mentions and web/url links into special tokens @USER and HTTPURL) with internal BERTweet normalizer. Also, we kept only two categories: hate speech (1) - 46021 tweets and not hate specch (0) - 80624 tweets. Then, we split the data into train, dev, test with following size:

`Train data size: 101316`
`Test data size: 12665`
`Dev data size: 12664`

The bertweet-base model was run for 3 epochs. With this model we manage to get:

`Precision Score: 81.576`
`Recall Score: 77.825`
`F1 Score: 79.657`

---

### Challenges

#### BERTweet 

For the baseline with BERTweet, we decided to use transfer learning by training the entire pre-trained BERTweet model on our dataset. Training pre-trained `vinai/bertweet-base` on training set of 101 316 tweets was compute-intensive. As a result, we did training with only 3 epochs to get the baseline results.


### Evaluation

We use F1-scrore as the major metric to evaluate our model. We compare the F1-scores of different models on cross-platform unseen data. The one which gives the best score is the best-suited for classification of generalized social-media text

---

### Conclusion

Our goal materializes from the fact that social media, being a widely used mode to socialize, has become unsafe for people looking for a secure environment to communicate. We come up with an efficient Deep Learning model to detect hate speech in online social media domain data using by fine tuning different variations of BERT pretrained model. This will become a useful tool to filter out any offensive and detrimental content across the social media platforms, even the ones which our model has never seen, and safeguard people from usage of hate speech.
