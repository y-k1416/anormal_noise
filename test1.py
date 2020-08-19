import os
import glob
import csv
import re
import itertools
import sys

import numpy as np
from tqdm import tqdm
from sklearn import metrics
import data_gen
import keras
from keras.models import Model


# parameter_load
# load param
param = data_gen.yaml_load()


def save_csv(save_file_path,
             save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)


def list_for_test(target_dir,
                  ext='wav'):
    dir_path = os.path.abspath('{dir}/test/*.{ext}'.format(dir=target_dir, ext=ext))
    file_paths = sorted(glob.glob(dir_path))

    machine_id_list = sorted(list(set(itertools.chain.from_iterable(
        [re.findall('id_[0-9][0-9]', ext_id) for ext_id in file_paths]))))
    return machine_id_list


def test_file(target_dir,
              id_name,
              normal="normal",
              anomaly="abnomal",
              ext="wav"):

    normal_files = sorted(
        glob.glob("data//*.{ext}".format(dir=target_dir,
                                                                             dir_name='test',
                                                                             prefix_normal=normal,
                                                                             id_name=id_name,
                                                                             ext=ext)))
    normal_labels = np.zeros(len(normal_files))
    anomaly_files = sorted(
        glob.glob("{dir}/{dir_name}/*.{ext}".format(dir=target_dir,
                                                                              dir_name='test',
                                                                              prefix_anomaly=anomaly,
                                                                              id_name=id_name,
                                                                              ext=ext)))
    anomaly_labels = np.ones(len(anomaly_files))
    files = np.concatenate((normal_files, anomaly_files), axis=0)
    labels = np.concatenate((normal_labels, anomaly_labels), axis=0)
    #com.logger.info("test_file  num : {num}".format(num=len(files)))
    if len(files) == 0:
        print("no_wav_file!!")
    print("\n========================================")

    return files, labels


#################
# main test
#################
if __name__ == '__main__':

    os.makedirs(param["result_directory"], exist_ok=True)
     

    csv_lines = []

    for idx, target_dir in enumerate(dirs):
        print('\n====================')
        print("[{idx}/{total}] {dirname}".format(dirname=target_dir, idx=idx+1, total=len(dirs)))
        machine_type = os.path.split(target_dir)[1]

        print('==========MODEL LOAD============')
        model_file = "{model}/model_{machine_type}.hdf5".format(model="model_dir",
                                                                machine_type=machine_type)

        # load model
        model = keras.models.load_model(model_file)
        model.summary()

        # results by type
        csv_lines.append([machine_type])
        csv_lines.append(["id", "AUC", "pAUC"])
        performance = []

        machine_id_list = list_for_test(target_dir)

        for id_str in machine_id_list:
            # load test file
            test_files, y_true = test_file(target_dir, id_str)

            # setup anomaly score file path
            anomaly_score_csv = "{result}/anomaly_score_{machine_type}_{id_str}.csv".format(
                                                                                     result=param["result_directory"],
                                                                                     machine_type=machine_type,
                                                                                     id_str=id_str)
            anomaly_score_list = []

            print("\n============== BEGIN TEST FOR A MACHINE ID ==============")
            y_pred = [0. for k in test_files]
            for file_idx, file_path in tqdm(enumerate(test_files), total=len(test_files)):


                try:
                    data = sd.file_to_array(file_path,
                                                    n_mels=param["feature"]["n_mels"],
                                                    frames=param["feature"]["frames"],
                                                    n_fft=param["feature"]["n_fft"],
                                                    hop_length=param["feature"]["hop_length"],
                                                    power=param["feature"]["power"])

                    errors = np.mean(np.square(data - model.predict(data)), axis=1)
                    y_pred[file_idx] = np.mean(errors)
                    anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])
                except:
                    print("file broken!!: {}".format(file_path))

            # save anomaly score
            save_csv(save_file_path=anomaly_score_csv, save_data=anomaly_score_list)

            if mode:
                # append AUC and pAUC to lists
                auc = metrics.roc_auc_score(y_true, y_pred)
                p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=param["max_fpr"])
                csv_lines.append([id_str.split("_", 1)[1], auc, p_auc])
                performance.append([auc, p_auc])
                print("AUC : {}".format(auc))
                print("pAUC : {}".format(p_auc))

            print("\n============ END OF TEST FOR A MACHINE ID ============")

        # calculate averages for AUCs and pAUCs
        averaged_performance = np.mean(np.array(performance, dtype=float), axis=0)
        csv_lines.append(["Average"] + list(averaged_performance))
        csv_lines.append([])

    # output results
    result_path = "{result}/{file_name}".format(result=param["result_directory"], file_name=param["result_file"])
    print("AUC and pAUC results -> {}".format(result_path))
    save_csv(save_file_path=result_path, save_data=csv_lines)
