import argparse
import os
from matplotlib import pyplot as plt
import tensorflow as tf
import larq as lq
from utils import get_model, load_data



def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', default='gtsrb')
    parser.add_argument('-bs', type=int, default=32)
    parser.add_argument('-epoch', type=int, default=30)
    parser.add_argument('-sz', type=int, default=64)
    parser.add_argument('-save_dir', default='weight')
    return parser.parse_args()


def main():
    arg = get_arg()

    # create save folder
    if not os.path.exists(arg.save_dir):
        os.makedirs(arg.save_dir, exist_ok=True)

    # load train / val / test dataset
    tr_ds, val_ds, test_ds, n_data, n_class = load_data(arg.data, 
                                                        (arg.sz, arg.sz), 
                                                        arg.bs)

    # load BNN
    model = get_model((arg.sz, arg.sz, 3), n_class)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', 
                  metrics=['accuracy'])
    lq.models.summary(model)

    # print
    print('class num:', n_class)
    print('train num:', n_data['train'])
    print('val num:', n_data['val'])
    print('test num:', n_data['test'])
    print('=' * 50)

    # callback
    callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                    factor=0.1,
                                                    patience=5)

    # train
    history = model.fit(tr_ds, 
                        epochs=arg.epoch,
                        validation_data=val_ds,
                        callbacks=[callback])
    history = history.history

    # save and load
    with lq.context.quantized_scope(True):    # save binary model
        model.save(f'{arg.save_dir}/bin_last.h5')
    model.save(f'{arg.save_dir}/fp_last.h5')  # save fp32 model
    model = tf.keras.models.load_model(f'{arg.save_dir}/bin_last.h5')
    
    # plot training history
    print('plot history ...')
    plt.plot(history['loss'], label='train')
    plt.plot(history['val_loss'], label='val')
    plt.title('Loss')
    plt.legend()
    plt.xlabel('Epoch')
    #plt.xticks(range(arg.epoch), range(1, arg.epoch+1))
    plt.savefig(f'{arg.save_dir}/loss.png')
    plt.close()

    plt.plot(history['accuracy'], label='train')
    plt.plot(history['val_accuracy'], label='val')
    plt.title('Accuracy')
    plt.legend()
    plt.xlabel('Epoch')
    #plt.xticks(range(arg.epoch), range(1, arg.epoch+1))
    plt.savefig(f'{arg.save_dir}/acc.png')
    plt.close()

    # evaluate on the testing data
    print('=' * 50)
    print('Evaluate the Testing data ...')
    model.evaluate(test_ds)
    print('=' * 50)



if __name__ == '__main__':
    main()