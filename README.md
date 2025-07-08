# TCSinger 2: Customizable Multilingual Zero-shot Singing Voice Synthesis

#### Yu Zhang, Wenxiang Guo, Changhao Pan, Dongyu Yao, Zhiyuan Zhu, Ziyue Jiang, Yuhan Wang, Tao Jin, Zhou Zhao | Zhejiang University

PyTorch implementation of **[TCSinger 2 (ACL 2025)](https://arxiv.org/abs/2505.14910): Customizable Multilingual Zero-shot Singing Voice Synthesis**.

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2505.14910)
[![GitHub Stars](https://img.shields.io/github/stars/AaronZ345/TCSinger2?style=social)](https://github.com/AaronZ345/TCSinger2)

> **Note:** The full source code will be released in this repository soon.

Listen to audio samples on our **[demo page](https://aaronz345.github.io/TCSinger2Demo/)**.

---

## News
- **2025-07:** Source code for TCSinger 2 released!  
- **2025-05:** TCSinger 2 accepted at ACL 2025!

---

## Key Features
- **TCSinger 2** is a multitask, multilingual, zero-shot singing-voice synthesis (SVS) model supporting both style transfer and fine-grained style control via diverse prompts.  
- A novel **Blurred-Boundary Content Encoder** robustly models phoneme and note transitions, yielding smoother phrasing.  
- A **Custom Audio Encoder**, trained with contrastive learning, extracts styles from arbitrary prompts, while the **Flow-based Custom Transformer** (with Cus-MOE and F0 clues) boosts synthesis quality and style fidelity.  
- Experiments show consistent gains over baselines—subjectively and objectively—on **zero-shot style transfer, hierarchical style control, cross-lingual transfer, and speech-to-singing conversion**.

---

## Quick Start
Below is a minimal example for generating high-fidelity samples with TCSinger 2.

Clone this repo on a machine equipped with an NVIDIA GPU, CUDA, and cuDNN, then follow the steps below.

### Dependencies
Create and activate a suitable conda environment named `tcsinger2`:

```bash
conda create -n tcsinger2 python=3.10
conda install --yes --file requirements.txt
conda activate tcsinger2
```

### Multi-GPU Training
By default, the code uses all GPUs returned by `torch.cuda.device_count()`.  
To limit GPUs, set the environment variable `CUDA_DEVICES_AVAILABLE` before launching training.

---

## Train Your Own Model

### 1 · Data Preparation
1. Collect a singing dataset—e.g., [GTSinger](https://github.com/AaronZ345/GTSinger)—and feel free to add extra data annotated with alignment tools.  
2. Place `metadata.json` (fields: `ph`, `word`, `item_name`, `ph_durs`, `wav_fn`, `singer`, `ep_pitches`, `ep_notedurs`, `ep_types`) and `phone_set.json` (complete phoneme list) in the desired folder and update the paths in `preprocess/preprocess.py`.  
   *A reference `metadata.json` is provided in **GTSinger***.  
3. Extract F0 for each `.wav`, save as `*_f0.npy`, e.g. with **[RMVPE](https://github.com/Dream-High/RMVPE)**.  
4. Preprocess the dataset:

```bash
export PYTHONPATH=.
python preprocess/preprocess.py
```

*Tip: You may also convert your dataset directly to a `.csv` instead of using `metadata.json`.*

5. Compute mel-spectrograms:

```bash
python preprocess/mel_spec_48k.py --tsv_path data/new/data.tsv --num_gpus 1 --max_duration 20
```

6. Post-process:

```bash
python preprocess/postprocess_data.py
```

### 2 · Model Training
1. **Train the VAE module**  
```bash
python main.py --base configs/ae_singing.yaml -t --gpus 0,1,2,3,4,5,6,7
```

2. **Train the main TCSinger 2 model**  
```bash
python main.py --base configs/tcsinger2.yaml -t --gpus 0,1,2,3,4,5,6,7
```

*Notes*  
- Adjust the compression ratio in the config files (and related scripts).  
- Change the padding length in the dataloader as needed.  
- To train the **Custom Audio Encoder**, format data as in `ldm/data/joinaudiodataset_con.py`, set the trained VAE path in `ae_con.yaml`, and proceed with training.

### 3 · Inference
```bash
python scripts/test_sing.py
```
*Replace the checkpoint path and CFG coefficient as required. For speech inputs, modify the VAE accordingly.*

---

## Acknowledgements
Our implementation builds upon code from:  
[Make-An-Audio-3](https://github.com/Text-to-Audio/Make-An-Audio-3) 
[TCSinger](https://github.com/AaronZ345/TCSinger)

(See individual files for detailed attributions.)

---

## Citation
If you find this work useful, please cite:

```bib
@article{zhang2025tcsinger,
  title   = {TCSinger 2: Customizable Multilingual Zero-shot Singing Voice Synthesis},
  author  = {Zhang, Yu and Guo, Wenxiang and Pan, Changhao and Yao, Dongyu and Zhu, Zhiyuan 
             and Jiang, Ziyue and Wang, Yuhan and Jin, Tao and Zhao, Zhou},
  journal = {arXiv preprint arXiv:2505.14910},
  year    = {2025}
}
```

---

## Disclaimer
The technologies described herein **must not** be used to generate anyone’s singing without explicit consent, including but not limited to government leaders, political figures, and celebrities. Unauthorized use may violate copyright and portrait rights.

![visitors](https://visitor-badge.laobi.icu/badge?page_id=AaronZ345/TCSinger2)
