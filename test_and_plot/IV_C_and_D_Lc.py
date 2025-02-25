import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import interpolate, optimize
import sys
from plot_utils import get_CAM_results
sys.path.insert(
    0,
    r'PLEASE_EDIT\PMR'
    )
import train_models
import models
from dataset_processing import dataset_processing_module


# 0. initialize
def get_inputs_labels(args, noise):
    dataset_processing_method = getattr(
        dataset_processing_module,
        'final_output_dataset'
        )
    datasets = {}
    datasets['train'], datasets['test'] = dataset_processing_method(
            args
            ).data_prepare()
    dataloaders = {
        phase:
        torch.utils.data.DataLoader(
            datasets[phase],
            batch_size=1650,  # 1650 is the sample numbers of test set
            shuffle=(True if phase == 'train' else False),
            )
            for phase in ['train', 'test']
        }
    phase = 'test'
    for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):

        if noise:
            num_test_samples = inputs.shape[0]
            snr = 0
            for i in range(num_test_samples):
                inputs[i, :] = dataset_processing_module.add_noise_perSNR(
                    inputs[i, :].cpu(),
                    snr
                    )

    label_index = {}  # obtain the corresponding index for every label
    classesnum = 11
    for i_label in range(classesnum):
        label_index[i_label] = [
            index for index,
            value in enumerate(labels.tolist()) if value == i_label
            ]
    return inputs, labels, label_index


classesnum = 11
args = train_models.args_ini()
args.dataset_name = 'EccGear'
matplotlib.use('TkAgg')
save_data_path = r'D:\202403_CAMandCNN\1_writing_papers' \
    r'\1_manuscript\plot_codes\inputs_data.pkl'

# 1. load dataset
for i_noise in [False, True]:
    inputs_data = {
        'inputs': [],
        'labels': [],
        'label_index': []
        }
    inputs_data['inputs']
    inputs_data['inputs'], \
        inputs_data['labels'], \
        inputs_data['label_index'] = get_inputs_labels(args, i_noise)
    # with open(save_data_path, 'wb') as f:
    #     pickle.dump(inputs_data, f)

    # with open(save_data_path, 'rb') as f:
    #     inputs_data = pickle.load(f)

    inputs = inputs_data['inputs'].unsqueeze(1)
    labels = inputs_data['labels']
    label_index = inputs_data['label_index']

    # 2. load models
    for i_w_wo_CAM in ['woCAM', 'wCAM']:
        w_wo_CAM = i_w_wo_CAM
        color0 = '#719AAC' if w_wo_CAM == 'woCAM' else '#925EB0'  # '#72B063'
        for i_model in ['DCNN', 'ResNet', 'Inception', 'AlexNet',
                        'DRSN', 'WKN_Laplace']:
            args.model_name = i_model
            if args.model_name == 'DCNN':
                args.alpha_cam = 10
            elif args.model_name == 'ResNet':
                args.alpha_cam = 0.2
            elif args.model_name == 'Inception':
                args.alpha_cam = 5
            elif args.model_name == 'AlexNet':
                args.alpha_cam = 30
            elif args.model_name == 'DRSN':
                args.alpha_cam = 1
            elif args.model_name == 'WKN_Laplace':
                args.alpha_cam = 0.2
            elif args.model_name == 'TFN_Chirplet':
                args.alpha_cam = 50
            model_dir = r'PLEASE_EDIT\PMR' + \
                r'\checkpoint\alpha' + str(args.alpha_cam) + '_120_' + \
                args.model_name + '_EccGear' + '_' + w_wo_CAM
            curves = np.empty((0, 16))
            for i in range(20):
                model_path = model_dir + r'\rep' + str(i) + r'\model.pth'
                model = getattr(
                    models, args.model_name
                    )(
                        in_channel=1, out_channel=classesnum
                        )
                model.load_state_dict(
                    torch.load(model_path, map_location='cuda')
                    )

                # 3. run CAM
                # phase = 'test'
                target_layer = get_CAM_results.choose_layer(
                    args.model_name, model
                    )
                net_cam = get_CAM_results.GradCAM(
                    model, target_layer, use_cuda=True
                    )
                cam_results = net_cam(inputs, args.CAM_type)
                get_CAM_results.avoid_memory_leak(net_cam)
                curves_one_model = np.array(cam_results.cpu().detach())
                curves = np.vstack((curves, curves_one_model))

            mean = np.mean(curves, axis=0)
            std = np.std(curves, axis=0)
            q1 = np.percentile(curves, 25, axis=0)
            q3 = np.percentile(curves, 75, axis=0)
            freq = [i * 3200/(16-1) for i in range(16)]

            # 4. plotting
            font = {
                    'family': 'Times New Roman',
                    'color':  'black',
                    'weight': 'normal',
                    'size': 10
                    }

            fig = plt.figure(1)
            fig.set_size_inches(3, 1.8)
            ax = fig.add_axes([0.11, 0.20, 0.88, 0.795])  # left, bottom, width, height
            ax.fill_between(
                freq,
                q1, q3,
                # mean+std, mean-std,
                color=color0, alpha=0.5,
                edgecolor=None
                )
            line1 = ax.plot(
                freq,
                mean,
                color=color0,
                linewidth=1
                )

            # set axis limit
            ax.set_ylim(-0.0, 1.05)
            ax.set_xlim(freq[0], freq[-1])

            # plot focused areas
            beta = 0.3  # β of the saliency weight is defined the focused area
            hline_value = (1 - beta) * (
                np.max(mean) - np.min(mean)
                ) + np.min(mean)
            # find the points
            idx = np.where(np.diff(np.sign(mean - hline_value)))[0]
            x_crossings = []
            for i in range(len(idx)):
                f = interpolate.interp1d(
                    [freq[idx[i]], freq[idx[i]+1]],
                    [mean[idx[i]], mean[idx[i]+1]]
                )

                def func(x):
                    return f(x) - hline_value

                # use optimize.root to find the roots
                result = optimize.root(
                    func,
                    (freq[idx[i]] + freq[idx[i]+1]) / 2
                    )  # set the mean value as the ini
                x_crossings.append(result.x)
            ylim_min, ylim_max = plt.gca().get_ylim()
            # plot the lines of the focused areas
            [plt.axvline(
                x=x_cor,
                ymax=(hline_value-ylim_min)/(ylim_max-ylim_min),
                color='r',
                linewidth=1,
                linestyle='--',
                alpha=0.5
                ) for x_cor in x_crossings]
            xlim_min, xlim_max = plt.gca().get_xlim()
            for i in range(int(len(x_crossings) / 2)):
                xmin_v = (x_crossings[2*i][0]-xlim_min)/(xlim_max-xlim_min)
                xmax_v = (x_crossings[2*i+1][0]-xlim_min)/(xlim_max-xlim_min)
                plt.axhline(y=0.0 * (ylim_max-ylim_min),
                            xmin=xmin_v, xmax=xmax_v,
                            color='r',
                            linewidth=6,
                            alpha=0.5
                            )
            # axis annotation
            ax.set_xlabel(
                'Frequency (Hz)', color='k', fontdict=font, labelpad=0
                )
            ax.tick_params(axis='both', labelsize=10)
            for label in ax.get_xticklabels():
                label.set_fontname('Times New Roman')
            # ax.set_ylim(-0.0, 1.05)
            ax.set_yticks(np.array([0, 1]))
            ax.set_ylabel(
                'Normalized amplitude (-)',
                color='k', fontdict=font, labelpad=0
                )
            for label in ax.get_yticklabels():
                label.set_fontname('Times New Roman')

            section = 'IV_D' if i_noise else 'IV_C'
            fig_save_path = r'D:\202403_CAMandCNN\1_writing_papers' \
                            + r'\1_manuscript\plot_codes\GearEcc_fig' \
                            + '\\' + section + '_Lc_' \
                            + args.model_name + '_' \
                            + w_wo_CAM + '.pdf'
            # plt.savefig(fig_save_path, format='pdf')

            plt.show()
            pass
pass
