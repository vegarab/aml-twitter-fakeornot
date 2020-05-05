FOR /L %%N IN (1, 1, 5) DO (
  python models/dummy.py -x
  python models/cosine_similarity.py -x
  python models/knn.py -x
  python models/naive_bayes.py -x
  python models/random_forest.py -x
  python models/svm.py -x
  python models/lstm.py -x
)