# Physically Meaningful Regularization (PMR)
A kind of physical-knowledge-guided and interpretable CNNs for gear fault severity level (FSL) diagnosis.

## 1.	Statement
This method uses the class activation mapping (CAM) results of input spectra to regulate the training process of deep diagnostic models.
It can:

1) Improve the models' intepretability.
2) Enhance the models' anti-noise ability.

To apply this method and use the code, please cite:

J. Li, X.-B. Wang, H. Chen, and Z.-X. Yang, “Physical-Knowledge-Guided and Interpretable Deep Neural Networks for Gear Fault Severity Level Diagnosis,” _IEEE Transactions on Industrial Informatics_, Early Access, doi: 10.1109/TII.2025.3547004.

Available at: [https://doi.org/10.1109/tii.2025.3547004](https://doi.org/10.1109/tii.2025.3547004).

## 2.	Method and paper introduction
Here, a brief introduction of the proposed method and the published is described, as shown in the workflow below.
![Method and paper introduction](https://github.com/user-attachments/assets/0a9db2d0-ea79-4e9c-9d4b-11812821f272)


This method tries to diagnose the FSL of localized faults. The idea is that, considering the situation for the diagnosis of FSL that all samples actually come from the same fault type, the diagnostic model is supposed to care more about the same areas of the input spectra, if the rotating speed remains constant. So, we proposed a PMR term to regulate the training process. The PMR term, which represents how much the diagnostic models concentrate, is calculated from the average of the CAM results of the training samples.

In the paper, we used two dataset to validate our method. And finally, different deep models with PMR outperformed that without, in terms of both the interpretability and anti-noise ability.

<!-- 
![image](https://github.com/LeeJMJM/PMR/assets/93640564/d8186cbe-13c4-4736-baaa-144d21c18cc7) 

Spectrum as input

![image](https://github.com/LeeJMJM/PMR/assets/93640564/2e7ef798-eba3-4d7a-8cc4-f0b859106c69)

The focused areas are scattered (two models without PMR)

![image](https://github.com/LeeJMJM/PMR/assets/93640564/e58c3cd3-139e-4807-8a43-7a4279b4c70e)

The focused areas are concerntrated (two models with PMR)
-->

## 3.	Code instruction
Here, how to use the code to replicate the paper is introduced.

### 3.1 Before running the code
**Dataset:** We used two datasets, UM-GearEccDatase and the XJTUSpurgear dataset. You can download the data used in this paper at [Data used in this paper from UM-GearEccDatase](https://drive.google.com/file/d/1zNxpOZuNije8oOqHQX7HxxiOn9NmmRTB/view?usp=sharing) and [Data used in this paper from the XJTUSpurgear dataset](https://drive.google.com/file/d/10-3or-IHJWOh2au0cNP6Yngv1itVu6Fm/view?usp=sharing). Or, you can also download the data from the corresponding cited references. Than, you should put the downloaded data in the _dataset_ folder. In addition, you should cite the corresponding references in your paper to be published.

**Initialization:** We ran the codes in _train_models.py_ to train models. Bascially, all the hyperparameters or settings were controlled there, either in the function _args\_ini()_ or in the loops from the _if \_\_name\_\_ == '\_\_main\_\_'_ idiom.

### 3.2 Train models and get CAM results in every epoch
**Train models:** As forementioned, training begins in _train_models.py_, but most of works are done in _trainer.train_and_test()_, where the parameters optimization iterates. _trainer.train_and_test()_ is defined from _train_utils.py_ in the folder _utils_.

**Get CAM results:** Unlike many papers using CAM to the trained models after the training process, our method gets CAM results in every epoch to calculate the PMR term to regulate the training process. To fulfill it, we calculate the CAM results of a minibatch through vectorization. The vectorization is realizzed by obtaining activation and gradient tensors by hook functions in _train_utils.py_ and calculating CAM results in _get_CAM_results.py_. Most details are described in the code comments but we would like to emphaszie two things:
- First, you should define a layer in the models under analysis to obtain feature maps, and the layer is defined in _get_CAM_results.py_.
- Second, every time you've obtained CAM results in an epoch, you'd better remove the hook functions to avoid memory leak, and it is realized by running the function _avoid_memory_leak()_ in _train_utils.py_.

### 3.3 Test the models' anti-noise ability visualize the results
**Test the anti-noise ability:** The diagnostic accuracy of the test set without additional noise is obtained just after the training process in _train_utils.py_, and you can find the results in the folder _checkpoint_. To get the diagnostic accuracy of the test set with additional noise, please run _IV_D_noise_accuracy_three_noise.py_ and _V_noise_accuracy_three_noise.py_ in the folder _test_and_plot_. Remember to change the folder directories and select the noise types.

**Test the anti-noise ability:** Other visualization of the CAM results shown in the paper (such as Fig. 5 to Fig. 11), can be realized by running the codes of other files in the folder _test_and_plot_. Just remember to change the folder directories.

### 3.4 Running environment
We've tried the two environments below and both work. The newer one runs faster.

- **Environment one:** Python 3.11.9, CUDA 12.4, PyTorch 2.4.0, and Windows 11.
- **Environment two:** Python 3.11.9, CUDA 12.1, PyTorch 2.2.2, and Windows 11.

## 4.	Acknowledgement
Apart from the refferences cited in the paper, we would like to express our sincere appreciation to the following persons for their public documents or instruction.
1. Brother Zihao from Tongji Univ. for his toturial courses of CAM applications. [Toturial courses of CAM](https://www.bilibili.com/video/BV1Ke411g7gm/?spm_id_from=333.337.search-card.all.click&vd_source=8acef43c041b678cb057f182421c1565)
2. Xin Zhang and Chao He for their public codes of 1D-version of CAM for fault diagnosis. [Codes of 1D-CAM in GitHub](https://github.com/liguge/1D-Grad-CAM-for-interpretable-intelligent-fault-diagnosis)
3. An anonymous coder who shares the code of the DRSN based on others article and codes. [Codes of DRSN](https://blog.csdn.net/weixin_47174159/article/details/115409058)


