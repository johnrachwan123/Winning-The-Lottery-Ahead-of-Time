# Pointer Sentinel Mixture Models
Implementation based on (https://github.com/ml-lab/Pointer-Sentinel-Mixture-Model)

In order to run call src/main.py:

Example:

```
python src/main.py --pruning_limit 0.9 --prune_criterion EarlyCroP --prune_to 10 --epochs 80
python src/main.py --pruning_limit 0.7 --prune_criterion EarlyCroPStructured --prune_to 10 --epochs 80
```

### Arguments

The regular arguments for running are the following.

| **argument**          | **description**                                            |**type**|
|-----------------------|------------------------------------------------------------|-------|
| --prune_criterion     | The pruning criterion (CroP, EarlyCroP, CroPStructured, EarlyCroPStructured, UnstructuredRandom)           | str   |
| --pruning_limit       | Final desired pruning sparsity | float |
| --prune_to       | Amoun of epochs to train for before performing pruning (only applicable for Early Baselines) | float |
| --batch_size       | Batch size to be used during training | float |
| --lr       | learning rate to be used during training | float |
| --epochs       | Number of epochs to train for| float |
| --hidden       | Dimension of hidden layers| float |

