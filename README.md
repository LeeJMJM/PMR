# Physically Meaningful Regularization (PMR)
A kind of physical-knowledge-guided and interpretable CNNs for gear fault severity level (FSL) diagnosis.

## 1.	Statement
This method uses the class activation mapping (CAM) results of input spectra to regulate the training process of deep diagnostic models.
It can:

1) Improve the models' intepretability.
2) Enhance the models' anti-noise ability.

To apply this method and use the code, please cite:

J. Li, X.-B. Wang, H. Chen, and Z.-X. Yang, “Physical-Knowledge-Guided and Interpretable Deep Neural Networks for Gear Fault Severity Level Diagnosis,” _IEEE Transactions on Industrial Informatics_, to be published.

## 2.	Method and paper introduction
Here, a brief introduction of the proposed method and the published is described, as shown in the figure below.
![Method and paper introduction](https://github.com/user-attachments/assets/e9d467de-ba67-40b8-b8b4-3aee8ab90b93)

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


### 3.2 Train models and get CAM results in every epoch


### 3.3 Test the models' anti-noise ability visualize the results




## 4.	Acknowledgement
Apart from the refferences cited in the paper, we would like to express our sincere appreciation to the following persons for their public documents or instruction.
1. 同济子豪兄 for his toturial courses of CAM applications. [Toturial courses of CAM](https://www.bilibili.com/video/BV1Ke411g7gm/?spm_id_from=333.337.search-card.all.click&vd_source=8acef43c041b678cb057f182421c1565)
2. Xin Zhang and Chao He for their public codes of 1D-version of CAM for fault diagnosis. [Codes of 1D-CAM in GitHub](https://github.com/liguge/1D-Grad-CAM-for-interpretable-intelligent-fault-diagnosis)
3. An anonymous coder who shares the code of the DRSN based on others article and codes. [Codes of DRSN](https://blog.csdn.net/weixin_47174159/article/details/115409058)


