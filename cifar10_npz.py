"""【Colab 版新增】取代 tfds.load("cifar10", ...)：改讀老師提供的 cifar10.npz（PI 2026/10/16）。

原本 tfds 從多倫多大學主機下載，每秒約 36 KB，全班會卡住。
回傳和 tfds 一樣的 {'image', 'label'} 資料集，後面的程式不用改。
"""
import types

import numpy as np
import tensorflow as tf

DATA_FILE = 'cifar10.npz'  # 上傳到 Colab 預設資料夾 /content 的檔案


def load_cifar10(split, with_info=False):
    """split 可寫 'train'、'test'、'train[:90%]'、'train[90%:]'，用法和 tfds.load 相同"""
    data = np.load(DATA_FILE)
    name, _, rng = str(split).partition('[')  # 'train[:90%]' 拆成 'train' 與 ':90%]'
    x, y = data['x_' + name], data['y_' + name]
    if rng:  # 依百分比取一段，和 tfds 的寫法相同
        lo, hi = rng.rstrip(']').split(':')
        lo = int(lo.rstrip('%') or 0) * len(x) // 100
        hi = int(hi.rstrip('%') or 100) * len(x) // 100
        x, y = x[lo:hi], y[lo:hi]
    ds = tf.data.Dataset.from_tensor_slices({'image': x, 'label': y})
    if not with_info:
        return ds
    # 課本只用到 info 的這兩項：訓練資料筆數、類別名稱
    info = types.SimpleNamespace(
        splits={'train': types.SimpleNamespace(num_examples=len(data['x_train'])),
                'test': types.SimpleNamespace(num_examples=len(data['x_test']))},
        features={'label': types.SimpleNamespace(names=[str(s) for s in data['label_names']])})
    return ds, info
