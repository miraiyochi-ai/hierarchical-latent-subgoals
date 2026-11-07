"""Frozen V-JEPA 2 static-frame path and masked-mean T5 instruction features.

The two identical frames form one tubelet. This deliberately preserves H3's
per-image latent rather than claiming full-video V-JEPA evaluation.
"""
from __future__ import annotations
import torch
from torch.nn import functional as F

VISUAL_REPO = 'facebook/vjepa2-vitl-fpc64-256'
VISUAL_REV = 'b3c1679b7c34d3255ef3547f27c7b226aefab26f'
TEXT_REPO = 'google-t5/t5-base'
TEXT_REV = 'a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1'
VISUAL_DIM = 2048
TEXT_DIM = 768


def visual_model(cache_dir, device='cuda'):
    from transformers import VJEPA2Model
    model = VJEPA2Model.from_pretrained(VISUAL_REPO, revision=VISUAL_REV,
                                      cache_dir=str(cache_dir), attn_implementation='sdpa')
    return model.to(device).eval().requires_grad_(False)


def preprocess(frames, device):
    value = torch.as_tensor(frames, device=device)
    if value.dtype != torch.uint8 or value.ndim != 4 or value.shape[-1] != 3:
        raise ValueError('expected uint8 NHWC RGB')
    value = value.permute(0, 3, 1, 2).float() / 255.
    value = F.interpolate(value, (256, 256), mode='bilinear', align_corners=False)
    mean = value.new_tensor((.485, .456, .406))[None, :, None, None]
    std = value.new_tensor((.229, .224, .225))[None, :, None, None]
    return ((value - mean) / std).unsqueeze(1).repeat(1, 2, 1, 1, 1)


@torch.inference_mode()
def encode_images(model, frames):
    device = next(model.parameters()).device
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        tokens = model(pixel_values_videos=preprocess(frames, device), skip_predictor=True).last_hidden_state
    if tokens.shape[1:] != (256, 1024):
        raise ValueError(f'unexpected encoder tokens: {tokens.shape}')
    result = F.layer_norm(tokens.float().mean(1), (1024,))
    if not torch.isfinite(result).all():
        raise FloatingPointError('invalid visual feature')
    return result.cpu().numpy()


@torch.inference_mode()
def encode_instructions(texts, cache_dir, device='cuda'):
    from transformers import AutoTokenizer, T5EncoderModel
    tokenizer = AutoTokenizer.from_pretrained(TEXT_REPO, revision=TEXT_REV, cache_dir=str(cache_dir))
    model = T5EncoderModel.from_pretrained(TEXT_REPO, revision=TEXT_REV,
                                         cache_dir=str(cache_dir)).to(device).eval().requires_grad_(False)
    tokens = tokenizer(texts, padding=True, truncation=False, return_tensors='pt')
    if tokens['input_ids'].shape[1] > 128:
        raise ValueError('instruction exceeds fixed token budget; do not silently truncate')
    tokens = {k: v.to(device) for k, v in tokens.items()}
    hidden = model(**tokens).last_hidden_state.float()
    mask = tokens['attention_mask'].unsqueeze(-1)
    result = (hidden * mask).sum(1) / mask.sum(1)
    if result.shape != (len(texts), TEXT_DIM) or not torch.isfinite(result).all():
        raise ValueError('unexpected language feature')
    return result.cpu().numpy()
