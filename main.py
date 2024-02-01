import numpy as np
import pandas as pd 
from sklearn import feature_extraction, linear_model, model_selection, preprocessing

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
