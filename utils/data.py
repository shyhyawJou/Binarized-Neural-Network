import pandas as pd
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from os.path import exists
from tqdm import tqdm

import tensorflow as tf



def load_data(data_dir, shape, batch_size):
    # check
    if exists(f'{data_dir}/Train.csv') and exists(f'{data_dir}/Test.csv'):
        tr_csv = pd.read_csv(f'{data_dir}/Train.csv')
        test_csv = pd.read_csv(f'{data_dir}/Test.csv')
    else:
        raise ValueError(f'"Train.csv" or "Test.csv" is not found in "{data_dir}"')
    
    # load path, label
    tr_path, test_path = tr_csv['Path'].tolist(), test_csv['Path'].tolist()
    tr_path  = [f'{data_dir}/{path}' for path in tr_path]
    test_path  = [f'{data_dir}/{path}' for path in test_path]
    tr_label, test_label = tr_csv['ClassId'], test_csv['ClassId']

    # split
    (tr_path, val_path, 
     tr_label, val_label) = train_test_split(tr_path, 
                                             tr_label, 
                                             test_size=0.2, 
                                             random_state=42,
                                             stratify=tr_label)
    n_data = {'train': len(tr_path), 
              'val': len(val_path),
              'test': len(test_path)}

    # read image
    tr_ds = np.asarray([np.asarray(Image.open(path).resize(shape)) 
                        for path in tqdm(tr_path)])
    val_ds = np.asarray([np.asarray(Image.open(path).resize(shape)) 
                         for path in tqdm(val_path)])
    test_ds = np.asarray([np.asarray(Image.open(path).resize(shape)) 
                          for path in tqdm(test_path)])

    # create dataset
    tr_ds = tf.data.Dataset.from_tensor_slices((tr_ds, tr_label))
    tr_ds = tr_ds.cache().shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((val_ds, val_label))
    val_ds = val_ds.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = tf.data.Dataset.from_tensor_slices((test_ds, test_label))
    test_ds = test_ds.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return tr_ds, val_ds, test_ds, n_data, np.unique(test_label).size
