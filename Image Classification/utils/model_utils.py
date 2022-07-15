import importlib
import inspect
from typing import List

import torch.nn as nn
import torch.optim as opt

from utils.constants import *
from utils.data_loaders import *

models = {x: {} for x in types}

"""
Handles loading/init/saving of models and more
"""


def _read_all_class_names():
    """
    private function that imports all class references in a dictionary
    """

    for typ in types:
        for name in os.listdir(os.path.join(".", MODELS_DIR, typ)):
            if not ("__" in name) and ".py" in name:
                short_name = str(name.split(".")[0])
                short_name: str
                module = importlib.import_module(f"models.{typ}.{short_name}")
                class_reference = getattr(module, short_name)
                models[typ][short_name] = class_reference

    # hardcoded ones
    models[OPTIMS] = {}
    models[OPTIMS]["ADAM"] = opt.Adam
    models[OPTIMS]["RMSPROP"] = opt.RMSprop
    models[OPTIMS]["SGD"] = opt.SGD

    models[DATASETS] = {}
    models[DATASETS]["RUBBISH"] = get_rubbish_loaders
    models[DATASETS]["CIFAR10"] = get_cifar10_loaders
    models[DATASETS]["CIFAR100"] = get_cifar100_loaders
    models[DATASETS]["TINYIMAGENET"] = get_tiny_imagenet_loaders
    models[DATASETS]["IMAGENET"] = get_imagenet_loaders
    models[DATASETS]["IMAGENETTE"] = get_imagenette_loaders
    models[DATASETS]["IMAGEWOOF"] = get_imagewoof_loaders
    models[DATASETS]["MNIST"] = get_mnist_loaders
    models[DATASETS]["fashion"] = get_fashionmnist_loaders
    models[DATASETS]["OMNIGLOT"] = get_omniglot_loaders
    models[DATASETS]["SVHN"] = get_SVHN_loaders
    models[DATASETS]["KMNIST"] = get_kmnist_loaders


def find_right_model(type: str, name: str, **kwargs):
    """
    returns model with arguments given a string name-tag
    """

    return models[type][name](**kwargs)


def detach_list(items):
    for i, x in enumerate(items):
        items[i] = x.detach()


def delete_list(items):
    for i, x in enumerate(items):
        del items[i], x


def save_models(models: List[nn.Module],
                suffix: str):
    """
    Saves current state of models
    """

    for mod in models:
        modstr = str(mod.__class__).split(".")[-1].replace("'>", "_")
        state = mod.state_dict()

        DATA_MANAGER.save_python_obj(state,
                                     os.path.join(RESULTS_DIR, DATA_MANAGER.stamp, MODELS_DIR, modstr + suffix),
                                     print_success=False)


def calculate_accuracy(targets, output, *ignored):
    output = torch.nn.functional.softmax(output, dim=-1).detach()
    _, classifications = output.detach().max(dim=-1)
    return (targets.eq(classifications)).float().mean()


def assert_type(content, expected_type):
    """ makes sure type is respected"""

    assert_non_empty(content)
    func = inspect.stack()[1][3]
    assert isinstance(content, type(expected_type)), "No {} entered in {} but instead value {}".format(
        str(expected_type),
        func,
        str(content))


def assert_non_empty(content):
    """ makes sure not None or len()==0 """

    func = inspect.stack()[1][3]
    assert not (content is None), "Content is null in {}".format(func)
    if type(content) is list or type(content) == str:
        assert len(content) > 0, "Empty {} in {}".format(type(content), func)


import math
import numpy as np


def centering(K):
    n = K.shape[0]
    if K.shape == (1, 1):
        return K

    unit = np.ones([n, n])
    I = np.eye(n)
    H = I - unit / n

    return np.dot(np.dot(H, K),
                  H)  # HKH are the same with KH, KH is the first centering, H(KH) do the second time, results are the sme with one time centering
    # return np.dot(H, K)  # KH


def rbf(X, sigma=None):
    GX = np.dot(X, X.T)
    KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
    if sigma is None:
        mdist = np.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= - 0.5 / (sigma * sigma)
    KX = np.exp(KX)
    return KX


def kernel_HSIC(X, Y, sigma):
    return np.sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))


def linear_HSIC(X, Y):
    L_X = np.dot(X, X.T)
    L_Y = np.dot(Y, Y.T)
    return np.sum(centering(L_X) * centering(L_Y))


def linear_CKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = np.sqrt(linear_HSIC(X, X))
    var2 = np.sqrt(linear_HSIC(Y, Y))
    return hsic / (var1 * var2)


def batch_CKA(X_list, Y_list):
    CKA_minibatch = 0
    k = len(X_list)
    inv_k = 1 / k
    hsic = 0
    var1 = 0
    var2 = 0
    for X, Y in zip(X_list, Y_list):
        hsic += linear_HSIC(X, Y)
        var1 += linear_HSIC(X, X)
        var2 += linear_HSIC(Y, Y)
    return hsic / (np.sqrt(var1) * np.sqrt(var2))


def kernel_CKA(X, Y, sigma=None):
    hsic = kernel_HSIC(X, Y, sigma)
    var1 = np.sqrt(kernel_HSIC(X, X, sigma))
    var2 = np.sqrt(kernel_HSIC(Y, Y, sigma))

    return hsic / (var1 * var2)


def gram_linear(x):
    """Compute Gram (kernel) matrix for a linear kernel.

    Args:
      x: A num_examples x num_features matrix of features.

    Returns:
      A num_examples x num_examples Gram matrix of examples.
    """
    return x.dot(x.T)


def gram_rbf(x, threshold=1.0):
    """Compute Gram (kernel) matrix for an RBF kernel.

    Args:
      x: A num_examples x num_features matrix of features.
      threshold: Fraction of median Euclidean distance to use as RBF kernel
        bandwidth. (This is the heuristic we use in the paper. There are other
        possible ways to set the bandwidth; we didn't try them.)

    Returns:
      A num_examples x num_examples Gram matrix of examples.
    """
    dot_products = x.dot(x.T)
    sq_norms = np.diag(dot_products)
    sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
    sq_median_distance = np.median(sq_distances)
    return np.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))


def center_gram(gram, unbiased=False):
    """Center a symmetric Gram matrix.

    This is equvialent to centering the (possibly infinite-dimensional) features
    induced by the kernel before computing the Gram matrix.

    Args:
      gram: A num_examples x num_examples symmetric matrix.
      unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
        estimate of HSIC. Note that this estimator may be negative.

    Returns:
      A symmetric matrix with centered columns and rows.
    """
    if not np.allclose(gram, gram.T):
        raise ValueError('Input must be a symmetric matrix.')
    gram = gram.copy()

    if unbiased:
        # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
        # L. (2014). Partial distance correlation with methods for dissimilarities.
        # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
        # stable than the alternative from Song et al. (2007).
        n = gram.shape[0]
        np.fill_diagonal(gram, 0)
        means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
        means -= np.sum(means) / (2 * (n - 1))
        gram -= means[:, None]
        gram -= means[None, :]
        np.fill_diagonal(gram, 0)
    else:
        means = np.mean(gram, 0, dtype=np.float64)
        means -= np.mean(means) / 2
        gram -= means[:, None]
        gram -= means[None, :]

    return gram


def cka(gram_x, gram_y, debiased=False):
    """Compute CKA.

    Args:
      gram_x: A num_examples x num_examples Gram matrix.
      gram_y: A num_examples x num_examples Gram matrix.
      debiased: Use unbiased estimator of HSIC. CKA may still be biased.

    Returns:
      The value of CKA between X and Y.
    """
    gram_x = center_gram(gram_x, unbiased=debiased)
    gram_y = center_gram(gram_y, unbiased=debiased)

    # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
    # n*(n-3) (unbiased variant), but this cancels for CKA.
    scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

    normalization_x = gram_x.ravel().dot(gram_x.ravel())
    normalization_y = gram_y.ravel().dot(gram_y.ravel())
    return scaled_hsic / (np.sqrt(normalization_x) * np.sqrt(normalization_y))


def cka_batch(X_list, Y_list, debiased=False):
    """Compute CKA.

    Args:
      gram_x: A num_examples x num_examples Gram matrix.
      gram_y: A num_examples x num_examples Gram matrix.
      debiased: Use unbiased estimator of HSIC. CKA may still be biased.

    Returns:
      The value of CKA between X and Y.
    """
    scaled_hsic = 0
    normalization_x = 0
    normalization_y = 0
    for X, Y in zip(X_list, Y_list):
        gram_x = center_gram(gram_linear(X), unbiased=debiased)
        gram_y = center_gram(gram_linear(Y), unbiased=debiased)

        # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
        # n*(n-3) (unbiased variant), but this cancels for CKA.
        scaled_hsic += gram_x.ravel().dot(gram_y.ravel())

        normalization_x += gram_x.ravel().dot(gram_x.ravel())
        normalization_y += gram_y.ravel().dot(gram_y.ravel())
    return scaled_hsic / (np.sqrt(normalization_x) * np.sqrt(normalization_y))


def _debiased_dot_product_similarity_helper(
        xty, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x, squared_norm_y,
        n):
    """Helper for computing debiased dot product similarity (i.e. linear HSIC)."""
    # This formula can be derived by manipulating the unbiased estimator from
    # Song et al. (2007).
    return (
            xty - n / (n - 2.) * sum_squared_rows_x.dot(sum_squared_rows_y)
            + squared_norm_x * squared_norm_y / ((n - 1) * (n - 2)))


def feature_space_linear_cka(features_x, features_y, debiased=False):
    """Compute CKA with a linear kernel, in feature space.

    This is typically faster than computing the Gram matrix when there are fewer
    features than examples.

    Args:
      features_x: A num_examples x num_features matrix of features.
      features_y: A num_examples x num_features matrix of features.
      debiased: Use unbiased estimator of dot product similarity. CKA may still be
        biased. Note that this estimator may be negative.

    Returns:
      The value of CKA between X and Y.
    """
    features_x = features_x - np.mean(features_x, 0, keepdims=True)
    features_y = features_y - np.mean(features_y, 0, keepdims=True)

    dot_product_similarity = np.linalg.norm(features_x.T.dot(features_y)) ** 2
    normalization_x = np.linalg.norm(features_x.T.dot(features_x))
    normalization_y = np.linalg.norm(features_y.T.dot(features_y))

    if debiased:
        n = features_x.shape[0]
        # Equivalent to np.sum(features_x ** 2, 1) but avoids an intermediate array.
        sum_squared_rows_x = np.einsum('ij,ij->i', features_x, features_x)
        sum_squared_rows_y = np.einsum('ij,ij->i', features_y, features_y)
        squared_norm_x = np.sum(sum_squared_rows_x)
        squared_norm_y = np.sum(sum_squared_rows_y)

        dot_product_similarity = _debiased_dot_product_similarity_helper(
            dot_product_similarity, sum_squared_rows_x, sum_squared_rows_y,
            squared_norm_x, squared_norm_y, n)
        normalization_x = np.sqrt(_debiased_dot_product_similarity_helper(
            normalization_x ** 2, sum_squared_rows_x, sum_squared_rows_x,
            squared_norm_x, squared_norm_x, n))
        normalization_y = np.sqrt(_debiased_dot_product_similarity_helper(
            normalization_y ** 2, sum_squared_rows_y, sum_squared_rows_y,
            squared_norm_y, squared_norm_y, n))

    return dot_product_similarity / (normalization_x * normalization_y)


# needed to load in class references
_read_all_class_names()
