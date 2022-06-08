# Accelerated Sign Hunter: a Sign-based Black-box Attack via Branch-Prune Strategy and Stabilized Hierarchical Search

### Abstract
We propose the Accelerated Sign Hunter (ASH), a sign-based black-box attack under $l_\infty$ constraint. The proposed method searches an approximate gradient sign of loss *w.r.t.* the input image with few queries to the target model and crafts the adversarial example by updating the input image in this direction. It applies a Branch-Prune Strategy that infers the unknown sign bits according to the checked ones to avoid unnecessary queries. 
It also adopts a Stabilized Hierarchical Search to achieve better performance within a limited query budget. 
We provide a theoretical proof showing that the Accelerated Sign Hunter halves the queries without dropping the attack success rate (SR) compared with the state-of-the-art sign-based black-box attack.
Extensive experiments also demonstrate the superiority of our ASH method over other black-box attacks. In particular on Inception-v3 for ImageNet, our method achieves the SR of 0.989 with an average queries of 338.56, which is 1/4 fewer than that of the state-of-the-art sign-based attack to achieve the same SR.
Moreover, our ASH method is out-of-the-box since there are no hyperparameters that need to be tuned.

### About the paper
Illustration of our proposed Accelerated Sign Hunter:

<img src="figures/illustration.PNG#pic_center" width=800>

Results of our ASH compared to other black-box attacks:

<img src="figures/results.PNG#pic_center" width=450>

Visualization of adversarial examples crafted by ASH and its baseline:

<img src="figures/visualization.PNG#pic_center" width=800>

## Running the code
### Requirements
Our code is based on the following dependencies
- pytorch == 1.6.0
- torchvision == 0.7.0
- numpy == 1.19.5
- matplotlib == 2.0.0

We use Resnet-50, VGG-16, and Inpcetion-v3 pretrained with Pytorch as the target models.
To reproduce the results in our paper, run:
```sh
python main.py --model resnet --attack sh --loss ce --idxs idxs/resnet.npy
python main.py --model vgg --attack sh --loss ce --idxs idxs/vgg.npy
python main.py --model inception --attack sh --loss ce --idxs idxs/inception.npy
```
```sh
python main.py --model resnet --attack ash --loss cw --idxs idxs/resnet.npy
python main.py --model vgg --attack ash --loss cw --idxs idxs/vgg.npy
python main.py --model inception --attack ash --loss cw --idxs idxs/inception.npy
```
