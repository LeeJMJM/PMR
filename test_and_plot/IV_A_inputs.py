import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
sys.path.insert(
    0,
    r'PLEASE_EDIT\PMR'
    )
import train_models
sys.path.insert(
    0,
    r'PLEASE_EDIT\PMR\dataset_processing'
    )
import dataset_processing_module


# 0. initialize
def get_inputs_labels(args):
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

        # # add noise. comment it if unnecessary.
        # num_test_samples = inputs.shape[0]
        # snr = 0
        # for i in range(num_test_samples):
        #     inputs[i, :] = dataset_processing_module.add_noise_perSNR(
        #         inputs[i, :].cpu(),
        #         snr
        #         )
        pass

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
text_value = [
    'Label = {}'.format(i) for i in range(classesnum)
    ]
text_value.append('Average of all labels')


# 1. load dataset
inputs_data = {
    'inputs': [],
    'labels': [],
    'label_index': []
    }
inputs_data['inputs']
inputs_data['inputs'], \
    inputs_data['labels'], \
    inputs_data['label_index'] = get_inputs_labels(args)
with open(save_data_path, 'wb') as f:
    pickle.dump(inputs_data, f)

with open(save_data_path, 'rb') as f:
    inputs_data = pickle.load(f)

inputs = inputs_data['inputs']
labels = inputs_data['labels']
label_index = inputs_data['label_index']

for label_class in range(classesnum+1):
    if label_class != classesnum:
        # n = len(label_index[label_class])
        inputs_one_label = np.array(inputs[label_index[label_class]])
    else:
        inputs_one_label = np.array(inputs)
    mean = np.mean(inputs_one_label, axis=0)
    std = np.std(inputs_one_label, axis=0)
    q1 = np.percentile(inputs_one_label, 25, axis=0)
    q3 = np.percentile(inputs_one_label, 75, axis=0)
    freq = [i * 3200/512 for i in range(512)]

    # 2. plotting
    font = {
            'family': 'Times New Roman',
            'color':  'black',
            'weight': 'normal',
            'size': 10
            }
    color1 = 'k'

    fig = plt.figure(1)
    fig.set_size_inches(3, 1.82)
    ax = fig.add_axes([0.11, 0.20, 0.88, 0.795])  # left, bottom, width, height
    ax.fill_between(
        freq,
        q1, q3,
        color=color1, alpha=0.3,
        edgecolor=None
        )
    line1 = ax.plot(
        freq,
        mean,
        color=color1,
        linewidth=0.5
        )
    if label_class != classesnum:
        plt.text(2400, 5.5, text_value[label_class], fontdict=font)
    else:
        plt.text(1700, 5.5, text_value[label_class], fontdict=font)

    # gear meshing freq
    [ax.axvline(x=x_cor, color='r', linewidth=1, linestyle='-.', alpha=0.3)
     for x_cor in [j*582.8 for j in range(1, 5+1)]]

    # axis annotation
    ax.set_xlim(freq[0], freq[-1])
    ax.set_xlabel('Frequency (Hz)', color='k', fontdict=font,
                  labelpad=0)
    ax.tick_params(axis='both', labelsize=10)
    for label in ax.get_xticklabels():
        label.set_fontname('Times New Roman')
    ax.set_ylim(-1.2, 6.5)
    ax.set_ylabel('Normalized amplitude (-)', color='k', fontdict=font,
                  labelpad=0)
    for label in ax.get_yticklabels():
        label.set_fontname('Times New Roman')

    # fig_save_path = r'D:\202403_CAMandCNN\1_writing_papers\1_manuscript' \
    #                 r'\20240615\fig\IV_A_inputs_' + str(label_class) + '.pdf'

    # plt.savefig(fig_save_path, format='pdf')
    plt.show()
pass
