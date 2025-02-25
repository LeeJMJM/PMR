import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
import time
import ptitprince as pt
from matplotlib.font_manager import FontProperties
from plot_utils import get_CAM_results
sys.path.insert(
    0,
    r'PLEASE_EDIT\PMR'
    )
import train_models
import models
from dataset_processing import dataset_processing_module


# 0. initialize
def get_inputs_labels(args, snr):
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
            batch_size=585,  # 585 is the length of the test set
            shuffle=(True if phase == 'train' else False),
            )
            for phase in ['train', 'test']
        }
    phase = 'test'
    for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
        num_test_samples = inputs.shape[0]
        for i in range(num_test_samples):
            inputs[i, :] = dataset_processing_module.add_noise_perSNR(
                inputs[i, :].cpu(), snr, args.noise_type
                )

    label_index = {}  # get the index of every corresponding label
    for i_label in range(5):
        label_index[i_label] = [
            index for index,
            value in enumerate(labels.tolist()) if value == i_label
            ]
    return inputs, labels, label_index


def load_model_and_get_accuracy(args, inputs, labels, rep_number):
    '''
    calculate one model's accuracy in one SNR with all rep, both w. and wo.
    '''
    # load models
    for i_w_wo_CAM in ['woCAM', 'wCAM']:
        w_wo_CAM = i_w_wo_CAM
        if args.model_name == 'ResNet':
            args.alpha_cam = 50
        elif args.model_name == 'DCNN':
            args.alpha_cam = 1
        elif args.model_name == 'Inception':
            args.alpha_cam = 100
        elif args.model_name == 'DRSN':
            args.alpha_cam = 10
        elif args.model_name == 'WKN_Laplace':
            args.alpha_cam = 5
        model_dir = r'C:\Users\jashm\OneDrive\桌面\PMR' + \
            r'\checkpoint\alpha' + str(args.alpha_cam) + '_120_' + \
            args.model_name + '_XJTU_Spurgear_' + w_wo_CAM

        acc_all_rep_one_snr = []
        for i in range(rep_number):  # loop for all repetitions
            model_path = model_dir + r'\rep' + str(i) + r'\model.pth'
            model = getattr(
                models, args.model_name
                )(
                    in_channel=1, out_channel=5
                    )
            model.load_state_dict(
                torch.load(model_path, map_location='cuda')
                )
            model.cuda()

            # compute accuracy
            num_test_samples = inputs.shape[0]
            if args.model_name == 'AlexNet':
                model.eval()  # AlexNet has Dropout layers
            else:
                model.train()  # Other models have BN layers
            with torch.set_grad_enabled(False):  # No need for gradients
                correct = torch.eq(
                    model(inputs.cuda()).argmax(dim=1),
                    labels.cuda()
                    ).float().sum().item()
            accuracy = correct / num_test_samples * 100
            acc_all_rep_one_snr.append(accuracy)
        acc_all_rep_one_snr = np.array(acc_all_rep_one_snr)
        if i_w_wo_CAM == 'woCAM':
            acc_all_rep_one_snr_wo = acc_all_rep_one_snr
        else:
            acc_all_rep_one_snr_w = acc_all_rep_one_snr
    return acc_all_rep_one_snr_wo, acc_all_rep_one_snr_w


args = train_models.args_ini()
args.noise_type = 'Laplacian'  # 'white' or 'pink' or 'Laplacian'
args.dataset_name = 'XJTU_Spurgear'
matplotlib.use('TkAgg')
if args.noise_type == 'white':
    save_data_path = r'D:\202403_CAMandCNN\1_writing_papers' \
        r'\1_manuscript\plot_codes\acc_white_noise_XJTUSpurgear.pkl'
elif args.noise_type == 'pink':
    save_data_path = r'D:\202403_CAMandCNN\1_writing_papers' \
        r'\1_manuscript\plot_codes\acc_pink_noise_XJTUSpurgear.pkl'
elif args.noise_type == 'Laplacian':
    save_data_path = r'D:\202403_CAMandCNN\1_writing_papers' \
        r'\1_manuscript\plot_codes\acc_Laplacian_noise_XJTUSpurgear.pkl'
else:
    print('args.noise_type is not assigned!')
snr_list = [-4, -2, 0, 2, 4]
rep_number = 20
model_list = ['DCNN', 'ResNet', 'Inception',
              'DRSN', 'WKN_Laplace']
color1, color2 = '#719AAC', '#925EB0'


# 1. load dataset
acc_wo, acc_w = {}, {}
for i_model in model_list:
    acc_wo[i_model] = np.empty((rep_number, 0))
    acc_w[i_model] = np.empty((rep_number, 0))
for i_snr in snr_list:
    t0 = time.time()
    inputs_data = {
        'inputs': [],
        'labels': [],
        'label_index': []
        }
    inputs_data['inputs']
    inputs_data['inputs'], \
        inputs_data['labels'], \
        inputs_data['label_index'] = get_inputs_labels(
            args, i_snr)

    inputs = inputs_data['inputs'].unsqueeze(1)
    labels = inputs_data['labels']
    label_index = inputs_data['label_index']

    # 2. calculate accuracy
    for i_model in model_list:  # loop for models
        args.model_name = i_model
        # load models and calculate accuracy of this model
        # in one SNR for all rep, both w. and wo.
        acc_all_rep_one_snr_wo, acc_all_rep_one_snr_w = \
            load_model_and_get_accuracy(args, inputs, labels, rep_number)

        # save accuracy in all SNRs
        acc_wo[i_model] = np.hstack(
            (acc_wo[i_model], acc_all_rep_one_snr_wo.reshape(-1, 1))
            )
        acc_w[i_model] = np.hstack(
            (acc_w[i_model], acc_all_rep_one_snr_w.reshape(-1, 1))
            )
    print('One SNR costs {:.4f} s.'.format(time.time()-t0))
acc_noise = {'acc_wo': acc_wo, 'acc_w': acc_w}
with open(save_data_path, 'wb') as f:
    pickle.dump(acc_noise, f)

with open(save_data_path, 'rb') as f:
    acc_noise = pickle.load(f)

# 3. ploting
font = {
        'family': 'Times New Roman',
        'color':  'black',
        'weight': 'normal',
        'size': 10
        }


def adarray2df(snr_list, rep_number, acc_ndarray):
    df = {'snr': [], 'acc': []}
    for i_snr in range(len(snr_list)):
        for i_rep in range(rep_number):
            df['snr'].append(snr_list[i_snr])
            df['acc'].append(acc_ndarray[i_rep, i_snr])
    return df


for i_model in model_list:
    fig = plt.figure(1)
    fig.set_size_inches(3, 1.8)
    ax = fig.add_axes([0.146, 0.20, 0.85, 0.79])  # left, bottom, width, height

    # turn ndarray into pd.df for plotting
    acc_wo = acc_noise['acc_wo'][i_model]
    acc_w = acc_noise['acc_w'][i_model]
    df_wo = adarray2df(snr_list, rep_number, acc_wo)
    df_w = adarray2df(snr_list, rep_number, acc_w)

    pt.half_violinplot(
        data=df_wo, x="snr", y="acc",
        inner=None, linewidth=0, width=0.5, scale="count",
        color=color1, alpha=0.5
        )
    line1 = ax.plot(
        np.mean(acc_wo, axis=0),
        '.--',
        color=color1,
        label=r'${\textnormal{\fontfamily{ptm}\selectfont without }}'
              r'\mathcal{J}_{\mbox{\fontfamily{ptm}\selectfont{\scriptsize{PMR}}}}$'
        )
    print('Model: ', i_model, '(wo)')
    print(np.round(np.mean(acc_wo, axis=0), 2),
          '±', np.round(np.std(acc_wo, axis=0), 2))
    pt.half_violinplot(
        data=df_w, x="snr", y="acc",
        inner=None, linewidth=0, width=0.5, scale="count",
        color=color2, alpha=0.5
        )
    line2 = ax.plot(
        np.mean(acc_w, axis=0),
        '.-',
        color=color2,
        label=r'${\textnormal{\fontfamily{ptm}\selectfont with }}'
              r'\mathcal{J}_{\mbox{\fontfamily{ptm}\selectfont{\scriptsize{PMR}}}}$'
        )
    print('Model: ', i_model, '(w)')
    print(np.round(np.mean(acc_w, axis=0), 2),
          '±', np.round(np.std(acc_w, axis=0), 2))

    # axix notation
    ax.set_xlabel(
        'SNR (dB)', color='k', fontdict=font, labelpad=0
        )
    ax.tick_params(axis='both', labelsize=10)
    for label in ax.get_xticklabels():
        label.set_fontname('Times New Roman')
    ax.set_ylim(None, 101.5)
    ax.set_ylabel(
        'Accuracy (%)',
        color='k', fontdict=font, labelpad=0
        )
    for label in ax.get_yticklabels():
        label.set_fontname('Times New Roman')

    # legend
    # use TeX rendering
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        })

    lines = line1 + line2
    labels = [h.get_label() for h in lines]
    fontProp = FontProperties()
    fontProp.set_name('CMU serif')
    fontProp.set_size(10)
    plt.legend(lines, labels, loc='lower right', prop=fontProp)

    # end TeX rendering
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        })

    if i_model == 'DCNN':
        ax.set_yticks(np.array([90, 92, 94, 96, 98, 100]))

    fig_save_path = r'D:\202403_CAMandCNN\1_writing_papers\1_manuscript' \
                    + r'\plot_codes\XJTUSpurgear_fig' \
                    + '\\' + 'V_accuracySNR_' \
                    + i_model + '.pdf'
    # plt.savefig(fig_save_path, format='pdf')
    plt.show()

pass
