#!/bin/zsh

# Option
# 1 : ダウンロードしたILIデータのパス
# 2 : 前処理したデータのパス

# bash recipes/set_data.sh path_to_downloaded_data path_to_prepared_data

python prepare_data.py -p ${1} -op ${2}