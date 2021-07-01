from matplotlib.pyplot import winter
import numpy as np
import random
import torch
import os
import shutil
import pickle
import glob
import pandas as pd
import re
import json



def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class Logger():
    def __init__(self, save_path, name, args_cls=None, args_dict=None):
        """
        ログファイルを作り、変数のクラスを渡すことで変数を書き込む
        args:
            save_path : 出力ファイルをまとめる親ディレクトリのパス
            name : ファイル名。このクラスで後ろに時刻がつく
            args_cls : 変数をまとめているクラス。インスタンス変数にする必要あり
            args_dict : 変数をまとめている辞書。argpareseを使っている時は、こっちの方が便利
        """
        assert args_cls==None or args_dict==None, "there is no args passed to logger"
        # make 
        import datetime
        if not os.path.exists(save_path+"//log"):
            os.makedirs(save_path+"//log")
        
        dt_now = datetime.datetime.now().strftime('%Y_%m_%d_%H%M')
        filename = name+dt_now
        self.filepath = save_path+"//log//"+filename+".txt"

        f = open(self.filepath, "w")
        
        ## write args
        if args_cls!=None:
            dic = args_cls.__dict__
        else:
            dic = args_dict
        if len(dic) != 0:
            f.write("[ args ]\r\n")
        for key in dic:
            f.write("{} : {}\r\n".format(key, dic[key]))
        f.write("[ log ]\r\n")
        f.close()
    
    def __call__(self, msg):
        """
        msgは、辞書型にしておく.
        最初の要素はepochにするといいかも
        """
        f = open(self.filepath,"a")
        for key in msg:
            f.write("{} : {}, ".format(key, msg[key]))
        f.write("\r\n")
        f.close()

    def WriteMsg(self, string):
        """
        普通に文字列を書き込む
        args : string (str) : 書き込む文字列
        """
        f = open(self.filepath,"a")
        f.write(string)
        f.write("\r\n")
        f.close()


def set_directories(root, model_name, sub_dirs, force=False):
    """
    出力用のディレクトリを作成。
    同じ名前のモデルファイルがあったら止める。

    - args:
        - root (str) : path to root of output directory
        - model_name (str) : name of model
        - sub_dirs (list) : list of file names make under the model directory
        - force (bool) : If True, we will overwrite when same model_name is already exists
    
    example:
    ```set_directories("root", "model_name", ["plot", "configs", "model"])```

    --root
        |- model_name
        |   |      
        |   |- plot
        |   |- configs
        |   |- model
        |
        |- (some models)
    """
    if not os.path.isdir(root):
        os.makedirs(root)
    
    model_path = os.path.join(root, model_name)

    # 同じ名前のファイルがあったら止める
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    elif force==True:
        print("over write model")
        shutil.rmtree(model_path)
        os.mkdir(model_path)
    else:
        assert False, "this model name is already used"

    # サブディレクトリを作る
    for d in sub_dirs:
        tmp_path = os.path.join(model_path, d)
        if not os.path.isdir(tmp_path):
            os.mkdir(tmp_path)


def set_data(data_path, train_name="train", test_name="test"):
    """
    train, testデータに分割して保存
    
    - args:
        - data_path : データのルート
        - train_name : 
    """
    train_root = os.path.join(data_path, train_name)
    test_root = os.path.join(data_path, test_name)
    if not os.path.isdir(train_root):
        os.mkdir(train_root)
    if not os.path.isdir(test_root):
        os.mkdir(test_root)
    _files = os.listdir(data_path)
    files = [f for f in _files if os.path.isfile(os.path.join(data_path, f))]
    for file in files:
        f_path = os.path.join(data_path, file)
        df_tmp = pd.read_csv(f_path)
        data_len = df_tmp.shape[0]
        region = df_tmp["REGION"].unique()[0]
        if region == "Virgin Islands" or region == "Puerto Rico":# データ数がなんか違うやつ。
            continue
        # df_use = df_tmp[["unweighted_ili", "YEAR", "WEEK"]]
        df_use = df_tmp[["ili_ratio", "YEAR", "WEEK"]]
        numpy_use = df_use.to_numpy()
        train_num = data_len//3 * 2
        train_data = numpy_use[:train_num]
        test_data = numpy_use[train_num:]
        p_train = os.path.join(train_root, "{}.pkl".format(region))
        p_test = os.path.join(test_root, "{}.pkl".format(region))
        with open(p_train, "wb") as f:
            pickle.dump(train_data, f)
        with open(p_test, "wb") as f:
            pickle.dump(test_data, f)

def load_data(train_path_root, test_path_root):
    """
    データ読み込み

    - args:
        - train_path_root
        - test_path_root
    - return:
        - train_dic
            - key(region)
                - data (numpy.ndarray) : (N, column)
                - max (int) :
                - min (int)
        - test_dic
            - same as train
    """
    trian_dic = {}
    test_dic = {}
    file_names = os.listdir(train_path_root)
    for fn in file_names:
        region = re.findall(r"(.*)\.pkl",fn)[0]
        train_path_tmp = os.path.join(train_path_root, fn)
        test_path_tmp = os.path.join(test_path_root, fn)
        with open(train_path_tmp, 'rb') as f:
            train_data_tmp = pickle.load(f)
        with open(test_path_tmp, 'rb') as f:
            test_data_tmp = pickle.load(f)
        # get max min (リークになるので、testでもtrainのmaxminを使用する)
        tmp_max = train_data_tmp[:, 0].max()
        tmp_min = train_data_tmp[:, 0].min()
        tmp_train_lst = [train_data_tmp.astype("float32"), tmp_max, tmp_min]
        tmp_test_lst = [test_data_tmp.astype("float32"), tmp_max, tmp_min]
        trian_dic[region] = tmp_train_lst
        test_dic[region] = tmp_test_lst
    return trian_dic, test_dic

class ILIDataset(torch.utils.data.Dataset):
    """
    インフルエンザ用のデータセット。

    args:
        - data (dictionary) : list(data, max, min)
            - data (numpy.ndarrey) : 
            - max : 
            - min
    """
    def __init__(self, data, input_step, predict_step, use_clm, use_state):
        super().__init__()
        if use_state == "all":
            self.use_state = list(data.keys())
        else:
            self.use_state = use_state
        self.make_use_data(data, self.use_state)

        self.use_clm = use_clm
        self.predict_step = predict_step
        self.input_step = input_step
        self.keys = list(self.data.keys())
        self.start_max = self.data[self.keys[0]].shape[0] - (self.predict_step + self.input_step) + 1
        self.key_len = len(self.keys)
        self.datalen = self.start_max * len(self.keys)

        self.set_clm()
    
    def __len__(self):
        return self.datalen

    def __getitem__(self, idx):
        # idx がランダムで来るから、この実装はあんまりよくない。
        # key_idx = np.random.randint(0, self.key_len)
        # start_idx = np.random.randint(0, self.start_max)
        key_idx = idx // self.start_max
        start_idx = idx % self.start_max
        predict_idx = start_idx + self.input_step
        end_idx = predict_idx + self.predict_step

        key = self.keys[key_idx]
        x = self.data[key][start_idx:predict_idx, :]
        y = self.data[key][predict_idx-1:end_idx-1, :]
        tgt = self.data[key][predict_idx:end_idx, :]

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        tgt = torch.from_numpy(tgt)

        return x, y, tgt, key

    def set_clm(self):
        """使用するカラムのみにする"""
        for k in self.keys:
            self.data[k] = self.data[k][:, self.use_clm]
    
    def set_min_max(self):
        _min = 1000
        _max = -1000
        for k in self.keys:
            max_tmp = np.max(self.data[k])
            min_tmp = np.min(self.data[k])
            if min_tmp < _min:
                _min = min_tmp
            if max_tmp > _max:
                _max = max_tmp
        self.max_data = _max
        self.min_data = _min
    
    def make_use_data(self, data, use_state):
        """data : dictionary"""
        self.data = {}
        for us in use_state:
            tmp_data = data[us][0]
            tmp_max = data[us][1]
            tmp_min = data[us][2]
            tmp_data[:,[0]] = (tmp_data[:,[0]] - tmp_min) / (tmp_max - tmp_min)# Year, weekにはやらない
            tmp_data[:,[0]] = np.clip(tmp_data[:,[0]], 0, 1)
            self.data[us] = tmp_data

    def get_min_max(self):
        return [self.max_data, self.min_data]

def lr_func(step, d_model):
    """
    論文内のスケジューラー
    """
    warmup_steps = 5000
    step_term = min(step**0.5, step*warmup_steps**(-1.5))
    return d_model**0.5 * step_term


def save_config(path, config):
    path = os.path.join(path, "config.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=4)

def list2string(lst, split_char="_"):
    """
    リストの中身を展開して、文字列にする。

    args:
        - lst (list) : 展開するリスト
        - split_char : リストの中身を分割する文字(Default : "_")
    return:
        - txt (string) : 結合された文字列
    
    usege:

    ```py
    lst = [0, 1, 2, 3]
    list2string(lst, "_") # "0_1_2_3"
    ```
    """
    txt = "{}".format(lst[0])
    for l in lst[1:]:
        txt += "_{}".format(l)
    return txt

if __name__ == "__main__":
    print(list2string([0]))
