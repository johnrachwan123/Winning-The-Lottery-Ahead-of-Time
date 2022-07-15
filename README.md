
# Compression via Gradient Flow Preservation

[![Python 3.7](https://img.shields.io/badge/Python-3.7-3776AB.svg?logo=python)](https://www.python.org/) [![PyTorch 1.4](https://img.shields.io/badge/PyTorch-1.4-EE4C2C.svg?logo=pytorch)](https://pytorch.org/docs/1.4.0/) [![MIT](https://img.shields.io/badge/License-MIT-3DA639.svg?logo=open-source-initiative)](LICENSE)

This repository is the **official** implementation of [Winning the Lottery Ahead of Time: Efficient Early Network Pruning](https://proceedings.mlr.press/v162/rachwan22a.html) published at ICML 2022. In order to cite our work please use the following BibTex entry:

```reference
@InProceedings{pmlr-v162-rachwan22a,
  title = 	 {Winning the Lottery Ahead of Time: Efficient Early Network Pruning},
  author =       {Rachwan, John and Z{\"u}gner, Daniel and Charpentier, Bertrand and Geisler, Simon and Ayle, Morgane and G{\"u}nnemann, Stephan},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {18293--18309},
  year = 	 {2022},
  editor = 	 {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v162/rachwan22a/rachwan22a.pdf},
  url = 	 {https://proceedings.mlr.press/v162/rachwan22a.html},
  abstract = 	 {Pruning, the task of sparsifying deep neural networks, received increasing attention recently. Although state-of-the-art pruning methods extract highly sparse models, they neglect two main challenges: (1) the process of finding these sparse models is often very expensive; (2) unstructured pruning does not provide benefits in terms of GPU memory, training time, or carbon emissions. We propose Early Compression via Gradient Flow Preservation (EarlyCroP), which efficiently extracts state-of-the-art sparse models before or early in training addressing challenge (1), and can be applied in a structured manner addressing challenge (2). This enables us to train sparse networks on commodity GPUs whose dense versions would be too large, thereby saving costs and reducing hardware requirements. We empirically show that EarlyCroP outperforms a rich set of baselines for many tasks (incl. classification, regression) and domains (incl. computer vision, natural language processing, and reinforcment learning). EarlyCroP leads to accuracy comparable to dense training while outperforming pruning baselines.}
}
```
### Setup

- Install virtualenv

> `pip3 install virtualenv`

- Create environment

> `virtualenv -p python3 ~/virtualenvs/EarlyCroP`

- Activate environment

> `source ~/virtualenvs/EarlyCroP/bin/activate`

- Install requirements:

> `pip install -r requirements.txt`

- If you mean to run the 'Tiny-Imagenet' dataset: download and unpack in `/gitignored/data/`, then replace CIFAR10 with TINYIMAGENET below to run. Additional datasets can be added in a similar way (Imagewoof, imagenette, etc.)

### Image Classification
To reproduce Image Classification results refer to [Image Classification](https://github.com/johnrachwan123/Early-Cropression-via-Gradient-Flow-Preservation/tree/main/Image%20Classification)

### NLP
To reproduce NLP results on the PSMM network refer to the folder [NLP](NLP)

### Licence

[MIT Licence](LICENSE) 
