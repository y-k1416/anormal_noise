import os
import glob
import sys
from tqdm import tqdm
import yaml

import numpy as np
# import data_gen
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Activation


# load yml
def yaml_load():
    with open("./param.yaml") as stream:
        param = yaml.safe_load(stream)
    return param

# visualize
class visualizer(object):
    def __init__(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.fig = self.plt.figure(figsize=(30, 10))
        self.plt.subplots_adjust(wspace=0.3, hspace=0.3)

    def loss_plot(self, loss, val_loss):
        """
        Plot loss curve.
        loss : list [ float ]
            training loss time series.
        val_loss : list [ float ]
            validation loss time series.
        return   : None
        """
        ax = self.fig.add_subplot(1, 1, 1)
        ax.cla()
        ax.plot(loss)
        ax.plot(val_loss)
        ax.set_title("Model loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(["Train", "Validation"], loc="upper right")

    def save_figure(self, name):
        """
        Save figure.
        name : str
            save png file path.
        return : None
        """
        self.plt.savefig(name)



#########
# main
#########
if __name__ == '__main__':
    # load param
    param = yaml_load()
    # make directory
    os.makedirs('model', exist_ok=True)

    # initialize visual
    visualizer = visualizer()


    # model_save
    model_file_path = "model/model_{name}.hdf5".format(name='norm')
    # historylog
    history_img = "model/history_{name}.png".format(name='norm')

    print("============== DATA_Loding ==============")
    train_data = np.load('./sound_data.npy')
    print(train_data.shape)

    # trainmodel
    print("=========MODEL_TRAINING=========")
    input_dim = 192
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(128)(input_layer)
    encoder = BatchNormalization()(encoder)
    encoder = Activation('relu')(encoder)
    encoder = Dense(128)(input_layer)
    encoder = BatchNormalization()(encoder)
    encoder = Activation('relu')(encoder)
    encoder = Dense(128)(input_layer)
    encoder = BatchNormalization()(encoder)
    encoder = Activation('relu')(encoder)

    encoder = Dense(8)(input_layer)
    encoder = BatchNormalization()(encoder)
    encoder = Activation('relu')(encoder)

    decoder = Dense(128)(encoder)
    decoder = BatchNormalization()(decoder)
    decoder = Activation('relu')(decoder)
    decoder = Dense(128)(encoder)
    decoder = BatchNormalization()(decoder)
    decoder = Activation('relu')(decoder)
    decoder = Dense(128)(encoder)
    decoder = BatchNormalization()(decoder)
    decoder = Activation('relu')(decoder)

    decoder = Dense(input_dim)(decoder)

    model = Model(input_layer, decoder)
    model.summary()

    model.compile(**param["fit"]["compile"])
    print(train_data.shape)
    history = model.fit(train_data,
                        train_data,
                        epochs=param["fit"]["epochs"],
                        batch_size=param["fit"]["batch_size"],
                        shuffle=param["fit"]["shuffle"],
                        validation_split=param["fit"]["validation_split"],
                        verbose=param["fit"]["verbose"])

    visualizer.loss_plot(history.history['loss'], history.history['val_loss'])
    visualizer.save_figure(history_img)
    model.save(model_file_path)
    print('================END TRAINING===================')
