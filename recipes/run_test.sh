#!/bin/zsh

# Option
# 1 : gpu num
# 2 : check state
# 3 : epoch num
# 4 : model path

# zsh recipes/run_test.sh -1 Delaware 199 model_stateXXXXXXXXXX

model_path="from_remote/model_multi_state_instep10_predstep2_clmnum0_d512_bs64_maxepc200_nenc4_ndec4_henc8_hdec8_ffh1024_prh1024_posth1024_dorpre0.0_dopost0.0_dormodel0.2_lr0.0001_schNo_lossmse"

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

for st in ${statelist[@]}
do
    python test.py -g ${1} \
                    -model ${model_path} \
                    -state ${st} \
                    -epn 60
done

