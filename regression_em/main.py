"""main function to test Regression EM algorithm."""
from model import RegressionEM
from sklearn.metrics import roc_auc_score
import numpy as np
import scipy.sparse as sp

if __name__ == "__main__":
    # making artificial data
    sample_size = 5000
    expo_dim = 100
    rel_dim = 300

    expo_feature = np.random.rand(sample_size, expo_dim)
    rel_feature = np.random.rand(sample_size, rel_dim)

    expo_param = np.random.rand(expo_dim) - 0.5
    rel_param = np.random.rand(rel_dim) - 0.5

    expo_probs = 1 / (1 + np.exp(- expo_feature @ expo_param))
    rel_probs = 1 / (1 + np.exp(- rel_feature @ rel_param))
    click_probs = expo_probs * rel_probs
    outcomes = np.array([1 if x >= 0.3 else 0 for x in click_probs])

    # Train Test split
    is_train = np.array([p >= 0.5 for p in np.random.rand(sample_size)])

    expo_train = expo_feature[is_train]
    rel_train = rel_feature[is_train]
    label_train = outcomes[is_train]

    expo_test = expo_feature[~is_train]
    rel_test = rel_feature[~is_train]
    label_test = outcomes[~is_train]

    feat_train = np.hstack([expo_train, rel_train])
    feat_test = np.hstack([expo_test, rel_test])

    # learning
    rem = RegressionEM(split_index=100, max_iter=10, class_weights='balanced', alpha=1)
    rem.fit(sp.csr_matrix(feat_train), label_train)
    print(roc_auc_score(label_test, rem.predict_proba(feat_test)))
