# Few-shot Learning + Interpolation for Classification in Low-resource Dialogue Systems

In this project, we implement a novel training framework to eliminate class-imabalance issues in a low-resource dialogue system like the Virtual Patient project. We combine the contrastive loss \[1\] with a 1-nearest-neighbor search to improve generalization for rare classes. Additionally, we combine it with a "mixup" \[2\] based KL divergence loss as a data-augmentation technique which also helps maintain performance on frequent classes.

We implement this with three underlying architectures:
* Text-CNN \[3\]
* Self-attention RNN \[4\]
* BERT \[5\]

# Requirements

All code was developed on Python 3.7. Additional requirements include:
* pytorch >= 1.4.0
* transformers >= 3.0.2 ([Link](https://huggingface.co/transformers/))
* pretrained BERT `bert-based-uncased` ([Link](https://huggingface.co/bert-base-uncased)) 
* FAISS toolkit for efficient nearest neighbor search ([Link](https://github.com/facebookresearch/faiss))

# Usage
* For fine-tuning hyperparameters, run: <code> bash run_gs.sh </code>. Logs will be saved in the file specified by <code>--validation-log</code>.
* For testing, run: <code> bash run_test.sh </code>.

# References

\[1\] Raia Hadsell, Sumit Chopra, and Yann LeCun, “Dimensionality reduction by learning an invariant mapping,” in 2006 IEEE Computer Society Conference on ComputerVision and Pattern Recognition (CVPR’06). IEEE, 2006.


\[2\] Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, andDavid Lopez-Paz,  “mixup: Beyond empirical risk min-imization,” in ICLR 2018.

\[3\] Yoon Kim, “Convolutional neural networks for sentence classification,” in EMNLP 2014.

\[4\] Zhouhan Lin, Minwei Feng, Cicero Nogueira dos Santos, Mo Yu, Bing Xiang, Bowen Zhou, and Yoshua Bengio, “A structured self-attentive sentence embedding,” in ICLR 2017.

\[5\] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova, “Bert: Pre-training of deep bidirec-tional transformers for language understanding,” in NAACL 2019.
