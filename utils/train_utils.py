#!/usr/bin/python
# -*- coding:utf-8 -*-

import os
import time
import warnings
import torch
from torch import nn
from torch import optim
import models
from utils import get_CAM_results
from dataset_processing import dataset_processing_module
from utils import additional_test


class train_utils():
    def __init__(self, args, save_dir, logger):
        self.args = args
        self.save_dir = save_dir
        self.logger = logger
        # Load and process the datasets
        self.dataset_processing_method = getattr(
            dataset_processing_module,
            'final_output_dataset')

    def generalization_test(self, phase):
        args = self.args
        with torch.set_grad_enabled(False):  # No need for gradients
            # anti-noise
            snr = args.snr
            accuracy_noised = additional_test.get_accuracy_for_noised(
                self.datasets, phase, args, snr, self.model)
        return accuracy_noised

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def avoid_memory_leak(self):
        del self.activations, self.gradients
        self.handle_forward.remove()
        self.handle_backward.remove()

    def setup(self, datasets):
        """
        Initialize the dataloader, model, loss, and optimizer
        """
        args = self.args

        # Check GPU available
        if not torch.cuda.is_available():
            warnings.warn("gpu is not available")

        #  Initialize the dataloader
        self.datasets = datasets
        self.dataloaders = {
            phase:
                torch.utils.data.DataLoader(
                    self.datasets[phase], batch_size=args.batch_size,
                    shuffle=(True if phase == 'train' else False),
                    pin_memory=True
                    )
            for phase in ['train', 'test']}

        # Define the model
        if args.dataset_name == 'EccGear':
            num_classes = 11
        elif args.dataset_name == 'XJTU_Spurgear':
            num_classes = 5
        inputchannel = 1
        self.model = getattr(
            models,
            args.model_name)(
                in_channel=inputchannel,
                out_channel=num_classes)
        self.model.cuda()

        # Define the losses
        self.loss_classification = nn.CrossEntropyLoss()
        self.loss_cam = nn.MSELoss()

        # Define the optimizer
        if args.opt == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr,
                                       momentum=args.momentum,
                                       weight_decay=args.weight_decay)
        elif args.opt == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr,
                                        weight_decay=args.weight_decay)
        else:
            raise Exception("optimizer not implement")

    def train_and_test(self):
        """
        Training and testing process
        """
        args = self.args
        logger = self.logger
        correct_batch = 0
        acc_train_rec, obj_train_rec, cost_cam_train_rec, \
            acc_test_rec, obj_test_rec, cost_cam_test_rec, \
            acc_noised_rec = ([] for _ in range(7))

        for epoch in range(args.max_epoch):
            logger.info('-'*5 + 'Epoch {}/{}'
                        .format(epoch, args.max_epoch - 1) + '-'*5)

            # Each epoch has a training phase and a test phase
            for phase in ['train', 'test']:
                epoch_phase_start = time.time()

                # Define some temp variables of all batches in a phase
                correct_batch_accum = 0
                obj_batch_accum = 0.0
                cost_cam_batch_accum = 0.0
                if epoch == (args.max_epoch - 1) and phase == 'test':
                    # to record the labels for the confusion matrix
                    pred_rec, labels_rec = [], []

                # Each phase has several batches
                for _, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    if phase == 'train':
                        self.model.train()
                    elif phase == 'test':
                        self.model.eval()

                    inputs = torch.unsqueeze(inputs, 1).cuda()
                    labels = labels.cuda()

                    # Do the learning process.
                    # And we also care about the gradient for Lc in test if
                    # we want to know the cost_CAM.
                    with torch.set_grad_enabled(
                                                phase == 'train'
                                                or
                                                (
                                                    phase == 'test'
                                                    and
                                                    args.CAM_loss_added)
                                                ):
                        # here the hooks are added
                        if args.CAM_loss_added:
                            target_layer = get_CAM_results.choose_layer(
                                args.model_name, self.model)
                            # register a hook during forward process
                            # to obatin the activations
                            self.handle_forward \
                                = target_layer.register_forward_hook(
                                    self.save_activation)
                            # register a hook during BP process
                            # to obatin the gradients
                            self.handle_backward \
                                = target_layer.register_full_backward_hook(
                                    self.save_gradient)

                        logits = self.model(inputs)
                        cost_classification = self.loss_classification(
                            logits, labels)

                        if args.CAM_loss_added:
                            # define the classes to be the greatest-value ones
                            yc, _ = torch.max(logits, dim=1)
                            self.model.zero_grad()
                            # use the backward_ones for vetorized operations
                            backward_ones = torch.ones_like(yc)
                            yc.backward(backward_ones, retain_graph=True)
                            net_cam = get_CAM_results.GradCAM()
                            # obtain the CAM results
                            Lc = net_cam(
                                args.CAM_type,
                                self.activations,
                                self.gradients)
                            Lc_mean = torch.mean(Lc, dim=0)
                            cost_cam = self.loss_cam(
                                Lc,
                                Lc_mean.expand(
                                    Lc.shape[0], -1)
                                    )
                        else:
                            cost_cam = torch.tensor(0).cuda()

                        cost_cam_weighted = cost_cam * args.alpha_cam
                        obj_fun = cost_classification + cost_cam_weighted
                        pred = logits.argmax(dim=1)
                        correct_batch \
                            = torch.eq(pred, labels).float().sum().item()
                        obj_batch = obj_fun.item() * args.batch_size
                        obj_batch_accum += obj_batch
                        cost_cam_batch = cost_cam.item() * args.batch_size
                        cost_cam_batch_accum += cost_cam_batch
                        correct_batch_accum += correct_batch

                        # Calculate the training info and train the model
                        if phase == 'train':
                            self.optimizer.zero_grad()
                            obj_fun.backward()
                            self.optimizer.step()

                        # To avoid memory leak
                        if args.CAM_loss_added is True:
                            self.avoid_memory_leak()

                        # Record predicted and true labels of the final epoch
                        if epoch == (args.max_epoch - 1) and phase == 'test':
                            pred_rec = pred_rec + pred.tolist()
                            labels_rec = labels_rec + labels.tolist()

                # Save the train and test information via each epoch
                sample_number = len(self.dataloaders[phase].dataset)
                obj_epoch = obj_batch_accum / sample_number
                cost_cam_epoch = cost_cam_batch_accum / sample_number
                epoch_acc = correct_batch_accum / sample_number

                # Test the generalization ability,
                # i.e. the anti-noise ability.
                if phase == 'test':
                    if args.additional_test:
                        accuracy_noised = self.generalization_test(phase)
                    elif not args.additional_test:
                        accuracy_noised = 0
                    else:
                        raise Exception("Wrong args: additional_test")

                # record info. of each epoch
                if phase == 'train':
                    obj_train_rec.append(obj_epoch)
                    cost_cam_train_rec.append(cost_cam_epoch)
                    acc_train_rec.append(epoch_acc)
                elif phase == 'test':
                    obj_test_rec.append(obj_epoch)
                    cost_cam_test_rec.append(cost_cam_epoch)
                    acc_test_rec.append(epoch_acc)
                    acc_noised_rec.append(accuracy_noised)

                logger.info('Epoch: {} {}-Obj: {:.4f} '
                            '(cost_classification: {:.4f}, '
                            'cost_cam_weighted: {:.4f}) {}-Acc: {:.4f}, '
                            'Cost {:.4f} sec.'
                            .format(
                                    epoch,
                                    phase,
                                    obj_epoch,
                                    obj_epoch-cost_cam_epoch*args.alpha_cam,
                                    cost_cam_epoch*args.alpha_cam,
                                    phase,
                                    epoch_acc,
                                    time.time()-epoch_phase_start
                                    ))

        # save the final model
        logger.info('save model after the epoch {}, '
                    'Acc {:.4f}, '
                    'noise-Acc {:.4f}'
                    .format(epoch,
                            epoch_acc,
                            accuracy_noised
                            ))
        torch.save(self.model.state_dict(),
                   os.path.join(self.save_dir, 'model.pth'))

        return acc_train_rec, obj_train_rec, cost_cam_train_rec, \
            acc_test_rec, obj_test_rec, cost_cam_test_rec, \
            pred_rec, labels_rec, acc_noised_rec
