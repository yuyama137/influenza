#!/bin/zsh

# Option
# 1 : gpu num
# 2 : max epoch
# 3 : lr
# 4 : loss_fn
# 5 : scheduling method

# bash recipes/run_train_single_state.sh 0 200 0.0001 mse

statelist=("Alabama" \
        "Alaska" \
        "Arizona" \
        "Arkansas" \
        "California" \
        "Colorado" \
        "Connecticut" \
        "Delaware" \
        "DistrictofColumbia" \
        "Georgia" \
        "Hawaii" \
        "Idaho" \
        "Illinois" \
        "Indiana" \
        "Iowa" \
        "Kansas" \
        "Kentucky" \
        "Louisiana" \
        "Maine" \
        "Maryland" \
        "Massachusetts" \
        "Michigan" \
        "Minnesota" \
        "Mississippi" \
        "Missouri" \
        "Montana" \
        "Nebraska" \
        "Nevada" \
        "NewHampshire" \
        "NewJersey" \
        "NewMexico" \
        "NewYork" \
        "NewYorkCity" \
        "NorthCarolina" \
        "NorthDakota" \
        "Ohio" \
        "Oklahoma" \
        "Oregon" \
        "Pennsylvania" \
        "RhodeIsland" \
        "SouthCarolina" \
        "SouthDakota" \
        "Tennessee" \
        "Texas" \
        "Utah" \
        "Vermont" \
        "Virginia" \
        "Washington" \
        "WestVirginia" \
        "Wisconsin" \
        "Wyoming")

for state in ${statelist[@]}; do
    echo $state
    python train_single_state.py -g ${1} \
                                -instep 10 \
                                -predstep 1 \
                                -uc 0 2 \
                                -bs 64 \
                                -me ${2} \
                                -lr ${3} \
                                -out ./checkout_single_state \
                                -dpre 0.0 \
                                -dpost 0.0 \
                                -d 512 \
                                -fh 1024 \
                                -hpre 1024 \
                                -hpost 1024 \
                                -us ${state} \
                                -loss ${4} \
                                -sch No
done
