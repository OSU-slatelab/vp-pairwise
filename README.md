# Few-shot Learning + Interpolation for Classification in Low-resource Dialogue Systems

In this project, we implement a novel training framework to eliminate class-imabalance issues in a low-resource dialogue system like the Virtual Patient project. We combine the contrastive loss \[1\] with a 1-nearest-neighbor search to improve generalization for rare classes. Additionally, we combine it with a "mixup" \[2\] based KL divergence loss as a data-augmentation technique which also helps maintain performance on frequent classes.

# Setup

All code was developed on Python 3.7. Additional requirements include:
* pytorch >= 1.4.0
* transformers >= 3.0.2 [Link](https://huggingface.co/transformers/)
* FAISS toolkit for efficient nearest neighbor [Link](https://github.com/facebookresearch/faiss)


\[1\] Raia Hadsell, Sumit Chopra, and Yann LeCun, “Dimensionality reduction by learning an invariant mapping,” in 2006 IEEE Computer Society Conference on ComputerVision and Pattern Recognition (CVPR’06). IEEE, 2006.


\[2\] Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, andDavid Lopez-Paz,  “mixup: Beyond empirical risk min-imization,” in the International Conference on Learning Representations, 2018.
