import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import feature_extraction, model_selection

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

count_vectorizer = feature_extraction.text.CountVectorizer()

train_vectors = count_vectorizer.fit_transform(train_df["text"])
test_vectors = count_vectorizer.transform(test_df["text"])

xgb_classifier = xgb.XGBClassifier(
    n_estimators=1000, objective='binary:logistic', tree_method='hist',
    max_depth=3, learning_rate=0.1, n_jobs=6, random_state=0)

scores = model_selection.cross_val_score(
    xgb_classifier, train_vectors, train_df["target"], cv=5, scoring="f1")
xgb_classifier.fit(train_vectors, train_df["target"])
print(np.mean(scores))

sample_submission = pd.read_csv(
    "data/sample_submission.csv")
sample_submission["target"] = xgb_classifier.predict(test_vectors)

sample_submission.to_csv("submissions/xgb/submission.csv", index=False)
