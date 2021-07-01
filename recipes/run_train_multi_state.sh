#!/bin/zsh

# Option
# 1 : gpu num
# 2 : predict step
# 2 : max epoch
# 3 : lr
# 4 : loss_fn
# 5 : scheduling method

# bash recipes/run_train_multi_state.sh 0 1 200 0.0001 mse

python train_multi_state.py -g ${1} \
                        -instep 10 \
                        -predstep ${2} \
                        -uc 0 \
                        -bs 64 \
                        -me ${3} \
                        -lr ${4} \
                        -out ./checkout/multi_state \
                        -dpre 0.0 \
                        -dpost 0.0 \
                        -d 512 \
                        -fh 1024 \
                        -hpre 1024 \
                        -hpost 1024 \
                        -loss ${5} \
