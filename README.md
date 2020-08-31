## COMP6208 Advanced Machine Learning: Twitter - Real or Not?

[Real or Not? NLP with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started/overview)

### Exploratory Data Analysis
[This exploratory data analysis](https://github.com/vegarab/aml-twitter-fakeornot/blob/master/docs/eda/eda.pdf)
of the *Real or Not? NLP with Disaster Tweets* Kaggle competition dataset looks
at possible underlying patterns in the textual data contained in tweets. The
analysis includes syntactical features such as punctuation, spelling mistakes
and word frequencies, and semantic features such as sentiment, emojis and
bigrams.  Further, we look at the top splits in a decision tree and perform
latent semantic analysis in an attempt to uncover lower-dimensional patterns.
The analysis reveals some underlying differences between the classes and
demonstrates how machine learning can be used on this classification problem.

### Machine Learning Report
[This project paper](https://github.com/vegarab/aml-twitter-fakeornot/blob/master/docs/report/main.pdf)
discusses the use of non-neural and neural methods for binary text
classification in Tweets. A range of different feature extractors, using latent
semantic analysis, are tested with a wide range of models. Bayesian
optimisation is used to optimise hyperparameters. Gradient Boosting with
pre-trained embeddings performs best out of the non-neural methods, while a
2-layer LSTM RNN produces the best neural cross-validated result. Additionally,
the BERT model is finetuned for classification, scoring better than all
previous models tested. 
