#!/usr/bin/python
# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


def plot_curves(i_rep, obj_train_rec, obj_test_rec,
                acc_train_rec, acc_test_rec,
                cost_cam_train_rec, cost_cam_test_rec,
                acc_noised_rec, time_elapsed, args, fig_num=1):
    '''
    plot the training loss and accuracy curves
    '''
    fig = plt.figure(fig_num)
    fig.set_size_inches(16, 8)
    max_epoch = args.max_epoch

    xAxis = list(range(len(obj_train_rec)))
    ax1 = plt.subplot(2, 3, 1)
    ax2 = ax1.twinx()

    line1 = ax1.plot(xAxis,
                     obj_train_rec,
                     label='Train obj',
                     color='royalblue',
                     ls='-.'
                     )
    line2 = ax1.plot(xAxis,
                     obj_test_rec,
                     label='Test obj',
                     color='royalblue'
                     )
    line3 = ax2.plot(xAxis,
                     acc_train_rec,
                     label='Train accuracy',
                     color='tomato',
                     ls='-.'
                     )
    line4 = ax2.plot(xAxis,
                     acc_test_rec,
                     label='Test accuracy',
                     color='tomato'
                     )

    ax1.set_xlabel('Epoch')
    ax1.set_xlim(-1, max_epoch)
    ax1.set_ylabel('Object function value', color='royalblue')
    ax2.set_ylabel('Accuracy', color='tomato')
    # ax1.set_ylim(-0.2, None)
    ax2.set_ylim(0, 1.05)
    lines = line1 + line2 + line3 + line4
    labels = [h.get_label() for h in lines]
    plt.legend(lines, labels, loc='right')

    ax = plt.subplot(2, 3, 2)
    line1 = ax.plot(
        xAxis,
        np.array(obj_train_rec) -
        np.array(cost_cam_train_rec) * args.alpha_cam,
        label='Train classification cost',
        color='k',
        ls='-.')
    line2 = ax.plot(
        xAxis,
        np.array(obj_test_rec) -
        np.array(cost_cam_test_rec) * args.alpha_cam,
        label='Test classification cost',
        color='k')
    ax.set_xlabel('Epoch')
    ax.set_xlim(-1, max_epoch)
    ax.set_ylabel('Cross-entropy/Classification cost', color='k')
    # ax.set_ylim(-0.2, None)
    lines = line1 + line2
    labels = [h.get_label() for h in lines]
    plt.legend(lines, labels, loc='right')

    ax = plt.subplot(2, 3, 5)
    line1 = ax.plot(
        xAxis, cost_cam_train_rec,
        label='Train cost_cam',
        color='k',
        ls='-.',
        )
    line2 = ax.plot(
        xAxis, cost_cam_test_rec,
        label='Test cost_cam',
        color='k'
        )
    ax.set_xlabel('Epoch')
    ax.set_xlim(-1, max_epoch)
    ax.set_ylabel('CAM cost', color='k')
    # ax.set_ylim(-0.2, None)
    lines = line1 + line2
    labels = [h.get_label() for h in lines]
    plt.legend(lines, labels, loc='right')

    ax = plt.subplot(2, 3, 3)
    line1 = ax.plot(xAxis, acc_noised_rec, label='Acc_noised', color='k')

    x_min, x_max = ax.get_xlim()
    plt.text(
        3*(x_max-x_min)/5+x_min, 1,
        '{:.2f} %'.format(acc_noised_rec[-1]*100))

    ax.set_xlabel('Epoch')
    ax.set_xlim(-1, max_epoch)
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel('Additional accuracy', color='k')
    # ax.set_ylim(-0.2, None)
    lines = line1
    labels = [h.get_label() for h in lines]
    plt.legend(lines, labels, loc='upper left')

    ax = plt.subplot(2, 3, 6)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    plt.text(1, 9, 'Cost: {:.2f} s'.format(time_elapsed))
    plt.text(1, 8, 'Model: {}'.format(args.model_name))
    plt.text(1, 7, 'rep: {}'.format(i_rep))


class ConfusionMatrix(object):
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1
        pass

    def summary(self):  # Precision, Recall, Specificity
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        # for i in range(self.num_classes):
        #     TP = self.matrix[i, i]
        #     FP = np.sum(self.matrix[i, :]) - TP
        #     FN = np.sum(self.matrix[:, i]) - TP
        #     TN = np.sum(self.matrix) - TP - FP - FN
        #     Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
        #     Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
        #     Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.

    def plot(self):
        plt.subplot(2, 3, 4)
        matrix = self.matrix
        plt.imshow(matrix, cmap=plt.cm.Blues)
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        plt.yticks(range(self.num_classes), self.labels)
        plt.colorbar()
        plt.xlabel('True labels')
        plt.ylabel('Predicted labels')
        plt.title('Confusion matrix')

        # adding statistics
        thresh = matrix.max() / 2  # threshold for character colors
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # note that here matrix[y, x] rather than matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y,
                         info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        # plt.show()
