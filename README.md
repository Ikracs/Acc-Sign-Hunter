# Accelerated Sign Hunter

This is the code for our paper "*Accelerated Sign Hunter: a Sign-based Black-box Attack via Branch-Prune Strategy and Stabilized Hierarchical Search*", which has been accepted by **ICMR, 2022**.

### Abstract
We propose the Accelerated Sign Hunter (ASH), a sign-based black-box attack under $l_\infty$ constraint. The proposed method searches an approximate gradient sign of loss *w.r.t.* the input image with few queries to the target model and crafts the adversarial example by updating the input image in this direction. It applies a Branch-Prune Strategy that infers the unknown sign bits according to the checked ones to avoid unnecessary queries. 
It also adopts a Stabilized Hierarchical Search to achieve better performance within a limited query budget. 
We provide a theoretical proof showing that the Accelerated Sign Hunter halves the queries without dropping the attack success rate (SR) compared with the state-of-the-art sign-based black-box attack.
Extensive experiments also demonstrate the superiority of our ASH method over other black-box attacks. In particular on Inception-v3 for ImageNet, our method achieves the SR of 0.989 with an average queries of 338.56, which is 1/4 fewer than that of the state-of-the-art sign-based attack to achieve the same SR.
Moreover, our ASH method is out-of-the-box since there are no hyperparameters that need to be tuned.

### About the paper
Illustration of our proposed Accelerated Sign Hunter:

<img src="figures/illustration.PNG#pic_center" width=600>

Results of our ASH compared to other black-box attacks:

<img src="figures/results.PNG#pic_center" width=600>

Visualization of adversarial examples crafted by ASH and its baseline:

<img src="figures/visualization.PNG#pic_center" width=600>

## Running the code
### Requirements
Our code is based on the following dependencies
- pytorch == 1.10.2
- torchvision == 0.11.3
- numpy == 1.21.5
- matplotlib == 3.5.3
- timm == 0.6.13

***NEW:*** We added models with various architectures provided by [timm](https://github.com/huggingface/pytorch-image-models) as target models.

Attack is only conducted on correctly classified images.
Use the following command to launch an SH/ASH attack:
```sh
python main.py --attack ATTACK --config CFG_FILE_PTH --model MODEL --dataset imaegent-1k \
--loss cw --batch_size 100 --gpu YOUR_GPU_ID --log_root logs --verbose --print_freq 100 --early_stop
```
