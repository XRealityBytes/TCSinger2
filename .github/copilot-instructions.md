# TCSinger2 AI Agent Instructions

## Project Overview
TCSinger2 is a multilingual zero-shot singing voice synthesis (SVS) system using diffusion models with custom audio encoders and mixture-of-experts (MoE) architecture. The system supports style transfer, multi-level style control, and cross-lingual synthesis.

## Architecture Components

### Core Pipeline
1. **VAE (Variational Autoencoder)**: `ldm/models/autoencoder1d.py` - Compresses mel-spectrograms to latent space
2. **Diffusion Model**: `ldm/models/diffusion/cfm1_audio.py` - Conditional Flow Matching (CFM) for generation
3. **Main Model**: `ldm/modules/diffusionmodules/tcsinger2.py` - TCSinger2 with Mixture-of-Experts
4. **Vocoder**: `vocoder/hifigan/` - Converts mel-spectrograms to audio

### Key Model Features
- **Blurred Boundary Content Encoder**: Handles phoneme/note boundary transitions
- **Custom Audio Encoder**: Uses contrastive learning for style extraction
- **Flow-based Custom Transformer**: With Cus-MOE and F0 enhancement
- **Multi-language Support**: 9 languages (Chinese, English, French, German, Italian, Japanese, Korean, Russian, Spanish)

## Essential Development Patterns

### Configuration Management
- All models use YAML configs in `configs/`: `ae_singing.yaml`, `tcsinger2.yaml`, `ae_con.yaml`
- Use `OmegaConf` for configuration parsing: `OmegaConf.load(config_path)`
- Model instantiation: `instantiate_from_config(config.model)`

### Training Workflow
```bash
# 1. Train VAE and duration predictor
python main.py --base configs/ae_singing.yaml -t --gpus 0,1,2,3

# 2. Train main TCSinger2 model  
python main.py --base configs/tcsinger2.yaml -t --gpus 0,1,2,3
```

### Data Pipeline
- **Input Format**: `metadata.json` with fields: `ph`, `word`, `item_name`, `ph_durs`, `wav_fn`, `singer`, `ep_pitches`, `ep_notedurs`, `ep_types`, `emotion`, `singing_method`, `technique`
- **Preprocessing**: `preprocess/preprocess.py` converts to TSV format
- **Mel Extraction**: `preprocess/mel_spec_48k.py` generates mel-spectrograms
- **F0 Processing**: Extract F0 as `*_f0.npy` files using RMVPE

### Model Components Integration
- **MoE Structure**: Language experts (4) + caption experts (4) in `tcsinger2.py`
- **Gumbel Softmax**: Used for expert selection with temperature control
- **Load Balancing**: MoE includes load balancing loss to ensure even expert usage
- **Multi-GPU**: Uses `torch.cuda.device_count()` by default, control with `CUDA_DEVICES_AVAILABLE`

## Critical Dependencies
- **External Models**: FLAN-T5 (`useful_ckpts/flan-t5-large`), HiFi-GAN (`useful_ckpts/hifigan`)
- **Flash Attention**: Optional but recommended (`flash_attn`)
- **PyTorch Lightning**: v1.9.0 for training orchestration

## Testing & Inference
- **Main Script**: `scripts/test_sing.py` 
- **Vocoder Integration**: `HifiGAN` class handles mel-to-audio conversion
- **Sampling**: Uses CFM sampler for generation from noise

## File Organization Conventions
- `ldm/`: Core diffusion model components
- `configs/`: YAML configuration files
- `preprocess/`: Data preparation scripts
- `vocoder/`: Audio generation modules
- `utils/`: Shared utilities and helpers
- `scripts/`: Inference and testing scripts

## Common Debugging Points
- **Path Issues**: Use absolute paths, check `normalize_path()` in preprocessing
- **GPU Memory**: Adjust `max_duration`, `batch_size` in configs
- **F0 Plotting**: Model logs F0 predictions vs ground truth every 5000 steps
- **Missing Files**: Preprocessing script logs missing audio files in red

## Development Commands
```bash
export PYTHONPATH=.  # Always set before running scripts
python preprocess/mel_spec_48k.py --tsv_path data/new/data.tsv --num_gpus 1
python scripts/test_sing.py  # Inference
```

When working with this codebase, always consider the multi-stage training pipeline, configuration-driven architecture, and multi-language support requirements.