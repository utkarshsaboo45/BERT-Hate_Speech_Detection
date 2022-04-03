## Hate Speech Detection in Online Social Media using Transfer Learning

### Introduction

1. Where you introduce the task/problem you will work on. This answers the question: ``What is the nature of the task?`` (e.g., sentiment analysis, machine translation, language generation, style transfer, etc.?). 

*Our Answer:*

Our aim is to train a model for hate-speech-detection multi-class-classification problem, where we plan to use a dataset with raw (social media) text to classify it among classes such as `hate_speech`, `offensive_language` and `neither`.

2. Please explain ``what the task entails`` (e.g., taking in as input a sequence of text from a source language and turning it into a sequence of sufficiently equivalent meaning in target language). 

*Our Answer:*

- As a part of our baseline experiments, we plan to use `BERT` in its vanilla form and train a model by freezing/updating BERT embeddings using transfer learning.
- Further, we will attempt to modify this vanilla BERT into a `BERT-CNN` architecture by using CNN layers on top of BERT frozen/unfrozen embeddings.
- Finally, we will use this transfer learning approach to adapt multi-lingual BERT embeddings for our task. 

### Motivation and Contributions

The interaction among users on different social media platforms generate a vast amount of data. Users of these platforms often indulge in detrimental, offensive and hateful behavior toward numerous segments of the society. While hate speech is not a recent phenomenon, the recent surge in online forums allows perpetrators to more directly target their victims. In addition, hate speech may polarize public opinion and hurt political discourse, with detrimental consequences for democracies. Therefore, the primary motivation for this project is an effort to devise an automated mechanism to detect and filter such speech and create a safer, more user-friendly environment for social media users.

- Detecting hate speech is a complicated task from the semantics point of view. Moreover,  when it comes to middle- and low-resource domains, the research in hate speech detection is almost insignificant due to the lack of labeled data. This has resulted in the emergence of bias in technology.
- To address this issue, we plan to apply a transfer learning approach for hate speech detection in low-resource domains.

### Data
- What kind of ``data`` will you be using? ``Describe the corpus``: genre, size, language, style, etc. Do you have the data? Will you acquire the data? How? Where will you ``store`` your data? 

__# TODO__

### Engineering

1. ``Computing infrastructure``

Our computing infrastructure includes our personal computers and Google Colab.

2. ``DL-NLP methods``

We will use transfer learning by fine-tuning the pre-trained BERT model on our dataset for hate speech classification.

3. ``Framework``

Currently, we plan to use `PyTorch`, `FastText`, and `BERT` models from the `HuggingFace` library. We may include more frameworks as the situation demands.


### Previous Works
- Refer to one or more projects that people have carried out that may be somewhat relevant to your project. This will later be expanded to some form of ``literature review``. For the sake of the proposal, this can be very brief. You are encouraged to refer to previous work here as a way to alert you to benefiting from existing projects. Provide links to any such projects.

*Our Answer:*
As the research grows in the field of Hate-speech detection on social media platforms (e.g., in SemEval-2019, one of the major tasks was classifying Twitter data as either hateful or not hateful), many researchers have increasingly shifted focus toward applying Deep Learning models for this task. As a basis for our project, we referred to the following two papers:

[A BERT-Based Transfer Learning Approach for Hate Speech Detection in Online Social Media](https://arxiv.org/pdf/1910.12574.pdf)

[Hate speech detection on Twitter using transfer learning](https://www.sciencedirect.com/science/article/abs/pii/S0885230822000110)

These papers show that multi-lingual models such as `XLM-RoBERTa` and `Distil BERT ` are largely able to learn the contextual information in tweets and accurately classify hate and offensive speech.


### Evaluation
- How will you ``evaluate`` your system? For example, if you are going to do MT, you could evaluate in ``BLEU``. For text classification, you can use ``accuracy`` and ``macro F1`` score. If your projects involves some interpretability, you could use ``visualization`` as a vehicle of deriving insights (and possibly some form of ``accuracy`` as approbriate).

*Our Answer:*
Currently, we plan to use F1-scrore as the major metric to evaluate our model. This may change if required.


### Conclusion
- You can have a very brief conclusion just summarizing the goal of the proposal. (2-3 sentences max).

*Our Answer:*

Our goal materializes from the fact that social media, being a widely used mode to socialize, has become unsafe for people looking for a secure environment to communicate. We hope that this project will become a useful tool to filter out offensive and detrimental content across the internet and safeguard people from usage of hate speech.

---
