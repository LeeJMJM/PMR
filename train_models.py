#!/usr/bin/python
# -*- coding:utf-8 -*-

import argparse
import os
import logging
from utils import train_utils
from utils import vis_utils
import matplotlib.pyplot as plt
import matplotlib
import time
import pandas as pd
import numpy as np
from dataset_processing import dataset_processing_module
matplotlib.use("Agg")  # to avoid errors during saving the PDF file


def args_ini():
    parser = argparse.ArgumentParser(description='Train')

    # basic parameters
    parser.add_argument('--model_name', type=str, default='DCNN',
                        help='the name of the model')
    parser.add_argument('--dataset_name', type=str, default='EccGear',
                        help='dataset name: EccGear or XJTU_Spurgear')
    parser.add_argument('--CAM_loss_added', type=bool, default=True,
                        help='is it added?')
    parser.add_argument('--CAM_type', type=str, default='GradCAM',
                        help='GradCAM/GradCAMPP/PFM')
    parser.add_argument('--alpha_cam', type=float, default=0.2,
                        help='the trade-off paramater of the PMR term')
    parser.add_argument('--raw_signal_or_FFT', type=str, default='FFT',
                        help='raw_signal or FFT, setting the input type')
    parser.add_argument('--length_dataset', type=str, default=120,
                        help='s, max = 60*10=600')
    parser.add_argument('--data_dir', type=str,
                        default=r'C:\Users\jashm\OneDrive\桌面\PMR\dataset',
                        help='the directory of the data')
    parser.add_argument('--normlizetype', type=str,
                        choices=['0-1', '1-1', 'mean-std'],
                        default='mean-std',
                        help='data normalization methods')
    parser.add_argument('--checkpoint_dir', type=str,
                        default=r'C:\Users\jashm\OneDrive\桌面\PMR\checkpoint',
                        help='the directory to save the model')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batchsize of the training process')

    # optimization information
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'],
                        default='adam',
                        help='the optimizer')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='the initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='the momentum for sgd')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='the weight decay')
    parser.add_argument('--max_epoch', type=int, default=50,
                        help='max number of epoch')
    parser.add_argument('--additional_test', type=bool, default=True,
                        help='to test more (e.g. noise) in test phase?')
    parser.add_argument('--snr', type=float, default=0,
                        help='noise level of the noise added in testset')
    parser.add_argument('--noise_type', type=str, default='white',
                        help='white or pink or Laplacian')

    args = parser.parse_args()
    return args


def run_a_model(i_rep, num_iter, args, save_dir, datasets):
    # Prepare the saving path for the model
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # set the logger
    logger = logging.getLogger('num_iter'+str(num_iter))
    logger.setLevel(logging.INFO)
    logFormatter = logging.Formatter("%(asctime)s %(message)s",
                                     "%m-%d %H:%M:%S")
    fileHandler = logging.FileHandler(os.path.join(save_dir, 'training.log'))
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    # save the args
    for k, v in args.__dict__.items():
        logger.info("{}: {}".format(k, v))

    trainer = train_utils.train_utils(args, save_dir, logger)
    trainer.setup(datasets)

    time_before = time.time()
    acc_train_rec, obj_train_rec, cost_cam_train_rec, acc_test_rec, \
        obj_test_rec, cost_cam_test_rec, pred_rec, labels_rec, \
        acc_noised_rec, \
        = trainer.train_and_test()
    time_elapsed = time.time() - time_before

    logger.info('Process {} is completed. Time for all epoch: {:.2f} s'
                .format(num_iter, time_elapsed))

    vis_utils.plot_curves(i_rep, obj_train_rec, obj_test_rec, acc_train_rec,
                          acc_test_rec, cost_cam_train_rec,
                          cost_cam_test_rec, acc_noised_rec,
                          time_elapsed, args, fig_num=num_iter)
    if args.dataset_name == 'EccGear':
        labels_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    elif args.dataset_name == 'XJTU_Spurgear':
        labels_name = ['0', '1', '2', '3', '4']
    CM = vis_utils.ConfusionMatrix(num_classes=len(labels_name),
                                   labels=labels_name)
    CM.update(pred_rec, labels_rec)
    CM.plot()
    plt.savefig(os.path.join(save_dir, 'vis.pdf'))
    rec_for_curve = pd.DataFrame({
        'obj_train': obj_train_rec, 'acc_train': acc_train_rec,
        'obj_test': obj_test_rec, 'acc_test': acc_test_rec,
        'cost_class_train': (
            np.array(obj_train_rec)
            - np.array(cost_cam_train_rec)*args.alpha_cam
            ).tolist(),
        'cost_class_test': (
            np.array(obj_test_rec)
            - np.array(cost_cam_test_rec)*args.alpha_cam
            ).tolist(),
        'cost_cam_train': cost_cam_train_rec,
        'cost_cam_test': cost_cam_test_rec,
        'acc_noised_rec': acc_noised_rec
                                })
    rec_for_CM = pd.DataFrame({'pred_rec': pred_rec, 'labels_rec': labels_rec})
    rec_for_curve.to_csv(
        os.path.join(save_dir, 'rec_for_curve.csv'), index=False)
    rec_for_CM.to_csv(
        os.path.join(save_dir, 'rec_for_CM.csv'), index=False)

    return acc_test_rec[-1], acc_noised_rec[-1], \
        cost_cam_train_rec[-1], cost_cam_test_rec[-1], time_elapsed


# run here to obtain the trained models
if __name__ == '__main__':
    args = args_ini()
    num_iter = 1
    rep = 5  # number of repetation of models
    for i_CAM in ['woCAM', 'wCAM']:  # ['woCAM', 'wCAM']
        if i_CAM == 'wCAM':
            args.CAM_loss_added = True
        else:
            args.CAM_loss_added = False
        # ['DCNN', 'ResNet', 'Inception', 'AlexNet',
        # 'DRSN', WKN_Laplace']
        for i_model in ['DCNN', 'ResNet', 'Inception', 'AlexNet',
                        'DRSN', 'WKN_Laplace']:
            args.model_name = i_model
            for i_dataset in ['EccGear', 'XJTU_Spurgear']:
                # ['EccGear', 'XJTU_Spurgear']
                if i_dataset == 'EccGear':
                    args.lr = 0.0005  # EccGear:0.0005
                    if args.model_name == 'DCNN':
                        alpha_cam_list = [10]  # EccGear:10
                    elif args.model_name == 'ResNet':
                        alpha_cam_list = [0.2]  # EccGear:0.2
                    elif args.model_name == 'Inception':
                        alpha_cam_list = [5]  # EccGear:5
                    elif args.model_name == 'AlexNet':
                        alpha_cam_list = [30]  # EccGear:30
                    elif args.model_name == 'DRSN':
                        alpha_cam_list = [1]  # EccGear: 1
                    elif args.model_name == 'WKN_Laplace':
                        alpha_cam_list = [0.2]  # EccGear:0.2
                elif i_dataset == 'XJTU_Spurgear':
                    args.lr = 0.003  # XJTU_Spurgear: 0.003
                    if args.model_name == 'DCNN':
                        alpha_cam_list = [1]  # XJTU_Spurgear: 1
                    elif args.model_name == 'ResNet':
                        alpha_cam_list = [50]  # XJTU_Spurgear:50
                    elif args.model_name == 'Inception':
                        alpha_cam_list = [100]  # XJTU_Spurgear:100
                    elif args.model_name == 'AlexNet':
                        alpha_cam_list = [10]  # JTU_Spurgear:10
                    elif args.model_name == 'DRSN':
                        alpha_cam_list = [10]  # XJTU_Spurgear: 10
                    elif args.model_name == 'WKN_Laplace':
                        alpha_cam_list = [5]  # XJTU_Spurgear: 5
                else:
                    raise Exception("Wrong i_dataset.")
                args.dataset_name = i_dataset
                for alpha_cam in alpha_cam_list:
                    args.alpha_cam = alpha_cam
                    sub_dir = 'alpha' + str(args.alpha_cam) + \
                        '_' + str(args.length_dataset) + \
                        '_' + args.model_name + \
                        '_' + args.dataset_name + \
                        '_' + i_CAM
                    save_dir = os.path.join(
                        args.checkpoint_dir,
                        sub_dir)

                    folder_list = os.listdir(args.checkpoint_dir)
                    folder_exist = any(
                        sub_dir in folder for folder in folder_list
                        )
                    acc_all_rep, acc_noised_all_rep, \
                        cost_cam_train_all_rep, \
                        cost_cam_test_all_rep, \
                        time_all_rep \
                        = ([] for i in range(5))

                    if folder_exist:
                        print('num_iter = {}, {} already exists'
                              .format(num_iter, sub_dir))
                    else:
                        # load the dataset
                        dataset_processing_method = getattr(
                            dataset_processing_module,
                            'final_output_dataset')
                        datasets = {}
                        datasets['train'], datasets['test'] = \
                            dataset_processing_method(
                                args
                                ).data_prepare()
                        for i_rep in range(rep):  # rep
                            subsub_dir = 'rep'+str(i_rep)
                            save_subdir = os.path.join(
                                save_dir,
                                subsub_dir)
                            acc_one_rep, acc_noised_one_rep, \
                                cost_cam_train_one_rep, \
                                cost_cam_test_one_rep, \
                                time_one_rep \
                                = run_a_model(
                                    i_rep,
                                    num_iter,
                                    args,
                                    save_subdir,
                                    datasets)
                            acc_all_rep.append(
                                acc_one_rep)
                            acc_noised_all_rep.append(
                                acc_noised_one_rep)
                            cost_cam_train_all_rep.append(
                                cost_cam_train_one_rep)
                            cost_cam_test_all_rep.append(
                                cost_cam_test_one_rep)
                            time_all_rep.append(
                                time_one_rep)
                            print('Model: {}'
                                  .format(num_iter))
                            num_iter += 1
                        some_record = pd.DataFrame(
                            {'accuracy': acc_all_rep,
                                'mean': [
                                    np.mean(acc_all_rep)
                                    ] * rep,
                                'std': [
                                    np.std(acc_all_rep)
                                    ] * rep,
                                'accuracy_noised': acc_noised_all_rep,
                                'mean_noised': [
                                    np.mean(acc_noised_all_rep)
                                    ] * rep,
                                'std_noised': [
                                    np.std(acc_noised_all_rep)
                                    ] * rep,
                                'cost_cam_tr': cost_cam_train_all_rep,
                                'mean_c_tr': [
                                    np.mean(cost_cam_train_all_rep)
                                    ] * rep,
                                'std_c_tr': [
                                    np.std(cost_cam_train_all_rep)
                                    ] * rep,
                                'cost_cam_te': cost_cam_test_all_rep,
                                'mean_c_te': [
                                    np.mean(cost_cam_test_all_rep)
                                    ] * rep,
                                'std_c_te': [
                                    np.std(cost_cam_test_all_rep)
                                    ] * rep,
                                'time': time_all_rep,
                                'mean_time': [
                                    np.mean(time_all_rep)
                                    ] * rep,
                                'std_time': [
                                    np.std(time_all_rep)
                                    ] * rep})
                        some_record.to_csv(
                            os.path.join(
                                        save_dir,
                                        'some_record_' +
                                        sub_dir +
                                        '.csv'
                                        ),
                            index=False)
    print('All completed')
