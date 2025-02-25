import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties


def get_PMR(root_path, sub_path, rep_path_list):
    PMRT = []
    for rep_path in rep_path_list:
        file_dir = root_path + sub_path + rep_path + r'\rec_for_curve.csv'
        df = pd.read_csv(file_dir)
        PMRT_a_model = df['cost_cam_test'].tolist()
        PMRT.append(PMRT_a_model)
    return np.array(PMRT)


def compute_both_bound(data):
    mean = np.mean(data, axis=0)
    std = np.std(data[:, -1])
    q1 = np.percentile(data, 25, axis=0)
    q3 = np.percentile(data, 75, axis=0)
    return mean, std, q1, q3


for modelName in ['DCNN', 'ResNet', 'Inception',
                  'DRSN', 'WKN_Laplace']:
    root_path = r'PLEASE_EDIT\PMR\checkpoint'
    if modelName == 'DCNN':
        alphaValue = '10'
    elif modelName == 'ResNet':
        alphaValue = '0.2'
    elif modelName == 'Inception':
        alphaValue = '5'
    elif modelName == 'DRSN':
        alphaValue = '1'
    elif modelName == 'WKN_Laplace':
        alphaValue = '0.2'
    sub_path_w_alpha0 = r'\alpha0_120_' + modelName \
        + '_EccGear_wCAM'
    sub_path_w_alphaNot0 = r'\alpha' + alphaValue \
        + '_120_' + modelName + '_EccGear_wCAM'
    fig_save_path = r'D:\202403_CAMandCNN\1_writing_papers\1_manuscript' \
                    r'\plot_codes\GearEcc_fig\IV_C_PMR_' + modelName + '.pdf'
    rep_path_list = [r'\rep'+str(i) for i in range(5)]

    PMRT_w_alpha0 = get_PMR(root_path, sub_path_w_alpha0, rep_path_list)
    PMRT_w_alphaNot0 = get_PMR(root_path, sub_path_w_alphaNot0, rep_path_list)

    PMRT_w_alpha0_mean, PMRT_w_alpha0_std, \
        PMRT_w_alpha0_Q1, PMRT_w_alpha0_Q3 \
        = compute_both_bound(PMRT_w_alpha0)
    PMRT_w_alphaNot0_mean, PMRT_w_alphaNot0_std, \
        PMRT_w_alphaNot0_Q1, PMRT_w_alphaNot0_Q3 \
        = compute_both_bound(PMRT_w_alphaNot0)

    epoch_list = list(range(1, PMRT_w_alpha0.shape[1] + 1))

    # initialize
    fig = plt.figure(1)
    fig.set_size_inches(3, 2.6)
    ax1 = fig.add_axes([0.165, 0.14, 0.71, 0.78])  # left, bottom, width, height
    font = {
        'family': 'Times New Roman',
        'color':  'black',
        'weight': 'normal',
        'size': 10
        }

    color1, color2 = '#719AAC', '#925EB0'
    ax1.fill_between(
        epoch_list,
        PMRT_w_alpha0_Q1, PMRT_w_alpha0_Q3,
        color=color1, alpha=0.5,
        edgecolor=None)
    line1 = ax1.plot(epoch_list, PMRT_w_alpha0_mean,
                     color=color1, linestyle='--',
                     linewidth=1,
                     label=r'$\alpha = 0$')
    ax2 = ax1.twinx()
    ax2.fill_between(
        epoch_list,
        PMRT_w_alphaNot0_Q1, PMRT_w_alphaNot0_Q3,
        color=color2, alpha=0.5,
        edgecolor=None)
    line2 = ax2.plot(epoch_list, PMRT_w_alphaNot0_mean,
                     color=color2,
                     linewidth=1,
                     label=r'$\alpha$ = ' + alphaValue)

    # use TeX rendering
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })

    # legend
    lines = line1 + line2
    labels = [h.get_label() for h in lines]
    fontProp = FontProperties()
    fontProp.set_name('CMU serif')
    fontProp.set_size(10)
    plt.legend(lines, labels, loc='best', prop=fontProp)

    # end TeX rendering
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
    })

    # set scientific notation for x-axis and y-axis
    ax1.ticklabel_format(style='sci', useMathText=True,
                         scilimits=(0, 0), axis='y')
    ax1.yaxis.get_offset_text().set_fontsize(10)
    ax1.yaxis.get_offset_text().set_fontname('Times New Roman')
    ax2.ticklabel_format(style='sci', useMathText=True,
                         scilimits=(0, 0), axis='y')
    ax2.yaxis.get_offset_text().set_fontsize(10)
    ax2.yaxis.get_offset_text().set_fontname('Times New Roman')

    # double y-axis notation
    # ax1.set_ylim(0.04, None)
    # ax2.set_ylim(0, None)
    ax1.set_xlim(epoch_list[0], epoch_list[-1])
    ax2.set_xlim(epoch_list[0], epoch_list[-1])
    ax1.tick_params(axis='y', labelsize=10, labelcolor=color1)
    for label in ax1.get_yticklabels():
        label.set_fontname('Times New Roman')
    ax2.tick_params(axis='y', labelsize=10, labelcolor=color2)
    for label in ax2.get_yticklabels():
        label.set_fontname('Times New Roman')
    ax1.tick_params(axis='x', labelsize=10)
    for label in ax1.get_xticklabels():
        label.set_fontname('Times New Roman')
    ax1.set_ylabel('Value (-)', color='k', fontdict=font,
                   labelpad=0)
    # ax2.set_ylabel('XX (-)', color=color2, fontdict=font, labelpad=0)
    ax1.set_xlabel('Epoch (-)', fontdict=font, labelpad=0)
    ax1.set_xticks(np.array([1, 25, 50]))

    # plt.savefig(fig_save_path, format='pdf')
    plt.show()
    pass
