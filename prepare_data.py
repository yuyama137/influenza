import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import util

def main():
    parser = argparse.ArgumentParser(description='set data')
    parser.add_argument('--data_path', '-p', type=str, help="path to ili dataset")
    parser.add_argument('--out_path', '-op', type=str, help="root of prepared data")
    args = parser.parse_args()
    df = pd.read_csv(args.data_path)

    # 数値データのないフロリダ州を削除
    df = df[df.REGION != "Florida"]

    df = df.astype({"ILITOTAL" : int, "TOTAL PATIENTS" : int})

    eps = 10e-9
    df["ili_ratio"] = df["ILITOTAL"] / (df["TOTAL PATIENTS"] + eps)

    df_use = df[["REGION", "YEAR", "WEEK", "ili_ratio"]]

    # to dictionary
    clm_lst = df_use["REGION"].unique()
    df_dict = {}
    for clm in clm_lst:
        df_tmp = df_use[df_use["REGION"] == clm]
        df_dict[clm] = df_tmp
    
    for k, df_ in df_dict.items():
        path = os.path.join(args.out_path, "{}.csv".format(k))
        df_.to_csv(path)
    
    # set pkl data
    util.set_data(args.out_path)

if __name__ == "__main__":
    main()