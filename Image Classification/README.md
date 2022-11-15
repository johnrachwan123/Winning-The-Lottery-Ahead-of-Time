### Training Examples & Results

Some examples of training the models from the paper. OneCycleLR is use to train sparse models as they have been shown to benefit from Learning Rate warmup.
The code is inspired by the following [repository](https://github.com/StijnVerdenius/SNIP-it)

#### Unstructured Pruning (EarlyCroP)

To run training for EarlyCroP our unstructured pruning algorithm - with a ResNet18 on CIFAR10, run one of the following:

```train
python3 main.py --model ResNet18 --data_set CIFAR10 --prune_criterion EarlyCroP --pruning_limit 0.98 --outer_layer_pruning --epochs 80
```

#### Structured Pruning (EarlyCroP-Structured)

To run training for EarlyCroP our unstructured pruning algorithm - with a VGG16 on CIFAR10, run one of the following:

```train
python3 main.py --model VGG16 --data_set CIFAR10 --prune_criterion EarlyCroPStructured --pruning_limit 0.88 --epochs 80 --batch_size 256 --eval_freq 195
```

### Visualization

Results and saved models will be logged to the terminal, logfiles in result-folders and in tensorboard files in the `/gitignored/results/` folder. To run tensorboard's interface run the following:

```
tensorboard --logdir ./gitignored/results/
```

The regular arguments for running are the following. **Additionally, there are some more found in utils/config_utils.py.**

| **argument**          | **description**                                            |**type**|
|-----------------------|------------------------------------------------------------|-------|
| --prune_criterion     | The pruning criterion from models/criterions          | str   |
| --pruning_limit       | Final desired pruning sparsity | float |
| --prune_to       | Amoun of epochs to train for before performing pruning (only applicable for Early Baselines) | float |
| --batch_size       | Batch size to be used during training | float |
| --lr       | learning rate to be used during training | float |
| --epochs       | Number of epochs to train for| float |
| --model | Which Neural Network Model to use | str
| --data_set | Which dataset to use (MNIST, CIFAR10, CIFAR100 etc.) | str
| --eval_freq | After how many batches should the model be evaluated | int
