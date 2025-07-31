import numpy as np

class_to_index = {"Neutral": 0, "Irrelevant": 1, "Negative": 2, "Positive": 3}
index_to_class = {v: k for k, v in class_to_index.items()}

def names_to_ids(names):
    return np.array([class_to_index.get(name) for name in names])

def ids_to_names(ids):
    return np.array([index_to_class.get(i) for i in ids])
