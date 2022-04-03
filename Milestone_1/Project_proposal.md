## Hate Speech Detection in Online Social Media using Transfer Learning

### Introduction

1. Where you introduce the task/problem you will work on. This answers the question: ``What is the nature of the task?`` (e.g., sentiment analysis, machine translation, language generation, style transfer, etc.?). 

*Our Answer:*

This problem is a hate speech classification in Online Social Media using Transfer learning.

2. Please explain ``what the task entails`` (e.g., taking in as input a sequence of text from a source language and turning it into a sequence of sufficiently equivalent meaning in target language). 

*Our Answer:*

In this project, we use pre-trained BERT embeddings which are trained on general corpus like Wikipedia and Book Corpus. We fine-tune this pre trained model on a few annotated social media dataset and later use the modified model for hatespeech detection in online social media domain. The basic system will take a sentence from social media as input and the output will be a class that would signify whether the sentence contained any form of hate speech or not. 

### Motivation and Contributions


1. ``What is the motivation for pursuing this project?`` (**this could be because the problem is ``socially motivated`` and/or ``remains unsolved``** (e.g., ``toxic`` and/or ``racist`` comments on social media, given their pervasively harmful impact).  

*Our Answer:*
The interactions among users on different social media platforms generate a vast amount of data. 
Users of these platforms often indulge in detrimental, offensive, and hateful behavior toward numerous segments of society.
While hate speech is not a recent phenomenon, the rise of online forums allows perpetrators to more direct targeting of their victims. 
In addition, hate speech may polarize public opinion and hurt political discourse, with detrimental consequences for democracies;
therefore, the primary motivation for this project is the effort to devise an automated mechanism to check for such speech.

2.  What do you hope your ``contribution`` will be? Here, you could aim at providing a ``better system`` than what exists (e.g., more robust MT), 
an application on new data (possibly within a new domain) (e.g., ``tweet intent and topic detection on COVID-19 data``), 
a system that delivers insights on a new topic (e.g., ``scale and sentiment in tweets in different location as to COVID-19``), etc. 

*Our Answer:*

Identifying hate speech is a complicated task from the semantics point of view. Moreover,  when it comes to middle- and low-resource languages, the research in hate speech detection is almost insignificant due to the lack of labeled data. This is resulting in the emergence of biasedness in technology.
To address this issue, we apply a transfer learning approach to hate speech detection for languages that lack labeled data.


### Data
- What kind of ``data`` will you be using? ``Describe the corpus``: genre, size, language, style, etc. Do you have the data? Will you acquire the data? How? Where will you ``store`` your data? 



### *Engineering:*
1. What ``computing infrastructure`` will you use? Personal computers? Google Colab? Google Cloud TPUs?

*Our Answer:*

We will use our personal computers and Google Colab for this project.

2. What ``deep learning of NLP (DL-NLP)`` methods will you employ? For example, will you do ``text classification with BERT?``, ``MT with attention-based BiLSTMs``, ``language generation with transformers``, etc.? 

*Our Answer:*

We will use transfer learning by training the pre-trained BERT multilingual model on our dataset for hate speech classification.

3. ``Framework`` you will use. Note: You *must* use PyTorch. Identify any ``existing codebase`` you can start off with. Provide links.

We will use PyTorch and BERT models from HuggingFace library.

*Our Answer:*

### *Previous Works (minimal):*
- Refer to one or more projects that people have carried out that may be somewhat relevant to your project. This will later be expanded to some form of ``literature review``. For the sake of the proposal, this can be very brief. You are encouraged to refer to previous work here as a way to alert you to benefiting from existing projects. Provide links to any such projects.

*Our Answer:*
As the research grows in this field (e.g., in SemEval-2019, one of the major tasks was classifying Twitter data as either hateful or not hateful), many researchers have increasingly shifted focus toward applying Deep Learning models to this task. As a basis for our project, we referred to these two projects:

[A BERT-Based Transfer Learning Approach for Hate Speech Detection in Online Social Media](https://arxiv.org/pdf/1910.12574.pdf)

This paper talks about a transfer learning approach using the pre-trained language model BERT learned on General English Corpus (no specific domain) to enhance hate speech detection on publicly available online social media datasets. They also introduce new fine-tuning strategies to examine the effect of different embedding layers of BERT in hate speech detection.

[Hate speech detection on Twitter using transfer learning](https://www.sciencedirect.com/science/article/abs/pii/S0885230822000110)

This paper shows that multi-lingual models such as xlm- roberta and distil-Bert are largely able to learn the contextual information in the tweets and accurately classify hate and offensive speech.



### *Evaluation:*
- How will you ``evaluate`` your system? For example, if you are going to do MT, you could evaluate in ``BLEU``. For text classification, you can use ``accuracy`` and ``macro F1`` score. If your projects involves some interpretability, you could use ``visualization`` as a vehicle of deriving insights (and possibly some form of ``accuracy`` as approbriate).

*Our Answer:*

To evaluate our proposed ap- proach, we use two publicly available datasets that have been annotated for racism, sexism, hate, or offensive content on Twitter.  (Can add more about the metric...)


### *Conclusion (optional):*
- You can have a very brief conclusion just summarizing the goal of the proposal. (2-3 sentences max).

*Our Answer:*

Our main goal is to come up with an efficient model that can detect hate speech in online social media domain data. Since, social media is a medium resource with not a lot of annotated data available, we use the approach of Transfer Learning to fine tune BERT model pre trained on General Corpus to be able to detect Hate Speech in Online Social Media without having to provide a huge annotated corpus of social media data.

