#! /bin/bash

python main.py --attack sh --config cfgs/sh.json --model resnet50 --dataset imagenet-1k --loss cw --batch_size 100 --gpu 0 --log_root logs --verbose --print_freq 100 --early_stop
python main.py --attack sh --config cfgs/sh.json --model vgg16 --dataset imagenet-1k --loss cw --batch_size 100 --gpu 0 --log_root logs --verbose --print_freq 100 --early_stop
python main.py --attack sh --config cfgs/sh.json --model inception_v3 --dataset imagenet-1k --loss cw --batch_size 100 --gpu 0 --log_root logs --verbose --print_freq 100 --early_stop

python main.py --attack ash --config cfgs/ash.json --model resnet50 --dataset imagenet-1k --loss cw --batch_size 100 --gpu 0 --log_root logs --verbose --print_freq 100 --early_stop
python main.py --attack ash --config cfgs/ash.json --model vgg16 --dataset imagenet-1k --loss cw --batch_size 100 --gpu 0 --log_root logs --verbose --print_freq 100 --early_stop
python main.py --attack ash --config cfgs/ash.json --model inception_v3 --dataset imagenet-1k --loss cw --batch_size 100 --gpu 0 --log_root logs --verbose --print_freq 100 --early_stop
