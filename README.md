# DARTS_MultiLevel_Pruning
### EEG-Based Sleep Stage Classification via Differentiable Architecture Search and Multi-Level Pruning
#### *By: Van-Duc Khuat∗1, Yue Cao∗2, and Wansu Lim∗1
##### 1.Department of Electrical and Computer Engineering, Sungkyunkwan, University, Republic of Korea, Suwon. 
###### 2.School of Cyber Science and Engineering and Shenzhen Research Institute, Wuhan University, Wuhan, 430072, China. 
## Abstract
![DARTS_MultiLevel_Pruning](imgs/DARTS_MultiLevel_Pruning.png)
Accurate classification of sleep stages from elec- troencephalogram (EEG) signals is a fundamental component of the assessment of affective and physiological state, with direct relevance to sleep quality, emotional regulation, and neurological health. However, practical deployment of deep learning–based sleep staging systems remains challenging due to substantial inter-subject variability in EEG signals and the high compu- tational cost of state-of-the-art (SOTA) models, particularly in wearable and clinical environments. In this work, a unified two- stage framework is proposed that jointly optimizes neural archi- tecture design and model compression through the integration of differentiable architecture search (DARTS) and multi-level pruning strategies. In Stage 1, DARTS is employed to discover compact but expressive architectures under the joint influence of a bidirectional data pruning mechanism, which removes noisy, imbalanced and ambiguous samples, and a differentiable operation pruning mechanism, which dynamically suppresses low-utility operations via temperature-controlled annealing. In Stage 2, the architecture obtained from Stage 1 is further refined by pruning the post-search filter based on information capacity and independence, reducing the number of convolutional filters by 50% while preserving the accuracy of the classification. Experimental results on the Sleep-EDF-20 and Sleep-EDF-78 datasets demonstrate that the proposed method achieves robust and competitive SOTA performance across both stages while sub- stantially reducing model complexity. In addition, the framework exhibits strong generalization under 5-fold cross-validation and enables real-time inference on resource-constrained hardware. These findings indicate that the proposed framework provides a practical and scalable solution for affective-aware sleep moni- toring across wearable, home-based, and clinical scenarios, while offering a general paradigm for resource-efficient neural architec- ture design in biosignal-driven affective computing applications.


We used two public datasets in this study:
- [Sleep-EDF-20](https://gist.github.com/emadeldeen24/a22691e36759934e53984289a94cb09b)
- [Sleep-EDF-78](https://physionet.org/content/sleep-edfx/1.0.0/)

After downloading the datasets, you can prepare them as follows:
```
cd prepare_datasets
python prepare_physionet.py --data_dir /path/to/PSG/files --output_dir edf_20_npz --select_ch "EEG Fpz-Cz"
```

## Training DARTS_MultiLevel_Pruning

Sleep stage classification using EEG signals remains challenging due to signal variability, label ambiguity, and over-parameterized architectures.  
In this work, we propose a **two-stage framework**:

- **Stage 1:** Differentiable architecture search with integrated data and operation pruning for robust sleep EEG modeling.

- **Stage 2:** Post-search filter pruning based on information capacity and independence.  

## Reproduce Results (Stage 1 & Stage 2)

- main_dart_bdp_20.ipynb — Sleep-EDF-20 (Stage 1 + Stage 2)

Prerequisites:

-The proposed framework comprises two sequential stages. Both stages were implemented in PyTorch (v2.4.0) on the NVIDIA GPU Cloud (NGC) platform, using CUDA 12.6 for GPU acceleration, and executed on a high-performance GPU supercomputing environment.


---

## Contact
Van-Duc Khuat  
Department of Electrical and Computer Engineering,  
Sungkyunkwan University, Suwon, Republic of Korea  
Email: duckv@g.skku.edu

Wansu Lim*  
Department of Electrical and Computer Engineering,  
Sungkyunkwan University, Suwon, Republic of Korea  
*Corresponding author- Email: wansu.lim@skku.edu

"# DARTS_MultiLevel_Pruning_TAC" 
"# DARTS_MultiLevel_Pruning_TNSRE" 
