import pandas as pd
from sklearn.cluster import DBSCAN
#Here starts the fun part
train = pd.read_csv("/Users/matteoppa/Projectfma/fma/data/fma_metadata/fma_train.csv")
db = DBSCAN(eps=10, min_samples= 5)
db.fit(train)