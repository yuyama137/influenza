# 再現実装 : "Deep Transformer Models for Time Series Forecasting: The Influenza Prevalence Case"

[論文](https://arxiv.org/abs/2001.08317)の再現実装を行った。

## 論文について

- Transformerを用いた時系列予測タスクを扱っている。
- アメリカCDCが提供しているインフルエンザの流行データを用いて実験を行った。
- LSTM, RNN, ARIMAなどの既存手法よりも良い精度で1週間先の流行を予測することができた。

## 手順

### データの準備

1. CDCの[サイト](https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html)より州ごとのインフルエンザの患者数データを取得(あるいは[ここ](https://drive.google.com/file/d/1pAaj9ZZDXr8dXH5L7Q1nNOF2cGuPRL6g/view?usp=sharing))
2. `bash recipes/set_data.sh path_to_downloaded_data`を実行

### 実行方法

#### 州ごとのデータを用いて学習をする時

`bash recipes/run_train_single_state.sh 0 200 0.0001 mse`

詳細は[ここ](recipes/run_train_single_state.sh)参照。

#### 全ての州を用いて学習をする時

`bash recipes/run_train_multi_state.sh 0 1 200 0.0001 mse`

詳細は[ここ](recipes/run_train_multi_state.sh)を参照。

## 結果

全ての州を用いて学習を行った際の1ステップ先の予測。

![](./img/multi_texas.png)

### ToDo

- schedulerの適用

## Reference

- [Wu, Neo, et al. "Deep transformer models for time series forecasting: The influenza prevalence case." arXiv preprint arXiv:2001.08317 (2020).](https://arxiv.org/abs/2001.08317)
