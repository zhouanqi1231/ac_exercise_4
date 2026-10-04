"""
Audio-visual emotion recognition baseline on RAVDESS (mteb/RAVDESS_AV).

Shared building blocks used by both train.py and infer.py:

  * load_ravdess / label_names         -- dataset loading
  * extract_frames / extract_audio     -- decode video frames & audio
  * RavdessAVDataset / AVCollator       -- torch Dataset + batching
  * AVEmotionModel                      -- frozen SigLIP2 (video) +
                                           frozen Wav2Vec2 (audio) + MLP head
  * make_split                          -- stratified / subject-independent split

Design
------
Video branch : SigLIP2 ViT-B/32 (google/siglip2-base-patch32-256), FROZEN.
               Per-frame image embedding -> temporal average pooling -> (B, Dv).
Audio branch : Wav2Vec2 base (facebook/wav2vec2-base), FROZEN.
               last_hidden_state -> masked temporal mean -> (B, Da).
Fusion       : concat([video, audio]) -> MLP -> num_classes logits.

Only the fusion MLP is trained; both backbones stay in eval() and require_grad=False.
"""
from __future__ import annotations

import os
import re
import random
import tempfile
from contextlib import nullcontext
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

import io
import numpy as np
import soundfile as sf
import librosa

from transformers import (
    AutoModel,
    AutoImageProcessor,
    AutoFeatureExtractor,
    Wav2Vec2Model,
)

DEFAULT_SIGLIP = "google/siglip2-base-patch32-256"
DEFAULT_WAV2VEC2 = "facebook/wav2vec2-base"
DATASET_ID = "mteb/RAVDESS_AV"
TARGET_SR = 16000  # wav2vec2 expects 16 kHz

# RAVDESS 7-part filename, e.g. 02-01-06-01-02-01-12.mp4 ; group(7) == actor id
_RAVDESS_RE = re.compile(r"(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})")


# load audio
def load_audio(audio_field, target_sr=TARGET_SR):
    """audio_field 是 Audio(decode=False) 返回的 {'path':..., 'bytes':...}"""
    if audio_field.get("bytes") is not None:
        wav, sr = sf.read(io.BytesIO(audio_field["bytes"]), dtype="float32")
    else:
        wav, sr = sf.read(audio_field["path"], dtype="float32")

    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return np.ascontiguousarray(wav, dtype=np.float32)

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_ravdess(split: str = "test", cache_dir: Optional[str] = None):
    """Load mteb/RAVDESS_AV. Only a `test` split exists (~1440 clips).

    Audio is resampled to 16 kHz; the video column is kept *undecoded*
    ({'path', 'bytes'}) so we control frame sampling ourselves (no torchcodec
    dependency required).
    """
    from datasets import load_dataset, Audio

    ds = load_dataset('/scratch/project_2020210/dataset', split=split)
    ds = ds.cast_column("audio", Audio(decode=False))
    try:
        from datasets import Video

        ds = ds.cast_column("video", Video(decode=False))
    except Exception:
        # Older datasets without a Video feature: leave the column as-is and
        # let extract_frames() figure out the type at runtime.
        pass
    return ds


def label_names(ds) -> List[str]:
    """Ordered list of the 8 emotion class names."""
    feat = ds.features["emotion"]
    names = getattr(feat, "names", None)
    if names is None:
        names = sorted({str(x) for x in ds["emotion"]})
    return list(names)


# ---------------------------------------------------------------------------
# Frame / audio decoding helpers
# ---------------------------------------------------------------------------
def _uniform_indices(n_total: int, n_take: int) -> List[int]:
    """Pick `n_take` frame indices evenly across `n_total` frames.

    If the clip has fewer frames than requested, indices are cycled so the
    output always has exactly `n_take` entries.
    """
    if n_total <= 0:
        return [0] * n_take
    if n_total < n_take:
        return [i % n_total for i in range(n_take)]
    step = n_total / float(n_take)
    return [min(int(step * i), n_total - 1) for i in range(n_take)]


def _resolve_video_path(video_field: Any) -> Tuple[str, Optional[str]]:
    """Return (path, tmpfile). tmpfile is set when bytes had to be written out."""
    if isinstance(video_field, str):
        return video_field, None
    if isinstance(video_field, dict):
        path = video_field.get("path")
        if path and os.path.exists(path):
            return path, None
        data = video_field.get("bytes")
        if data:
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp.write(data)
            tmp.flush()
            tmp.close()
            return tmp.name, tmp.name
    raise ValueError(f"Cannot resolve a video path from field of type {type(video_field)}")


def _decode_with_decord(path: str, num_frames: int) -> List[Image.Image]:
    import decord

    vr = decord.VideoReader(path, ctx=decord.cpu(0))
    idx = _uniform_indices(len(vr), num_frames)
    batch = vr.get_batch(idx).asnumpy()  # (N, H, W, 3) RGB uint8
    return [Image.fromarray(f).convert("RGB") for f in batch]


def _decode_with_opencv(path: str, num_frames: int) -> List[Image.Image]:
    import cv2

    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    order = _uniform_indices(total if total > 0 else num_frames, num_frames)

    grabbed = {}
    for i in sorted(set(order)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if ok:
            grabbed[i] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if not grabbed:  # fall back to sequential read (some codecs ignore POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frames_all = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames_all.append(frame)
        cap.release()
        order = _uniform_indices(len(frames_all), num_frames)
        out = [Image.fromarray(cv2.cvtColor(frames_all[i], cv2.COLOR_BGR2RGB)) for i in order]
    else:
        cap.release()
        out = [grabbed.get(i, next(iter(grabbed.values()))) for i in order]

    if not out:
        raise RuntimeError(f"OpenCV failed to decode frames from {path}")
    return out


def _frames_from_decoder(dec, num_frames: int) -> List[Image.Image]:
    """Handle an already-decoded VideoDecoder object (torchcodec/decord)."""
    idx = _uniform_indices(len(dec), num_frames)
    out = []
    for i in idx:
        frame = dec[i]
        arr = frame.numpy() if hasattr(frame, "numpy") else np.asarray(frame)
        if arr.ndim == 3 and arr.shape[0] in (1, 3):  # CHW -> HWC
            arr = np.transpose(arr, (1, 2, 0))
        out.append(Image.fromarray(arr.astype(np.uint8)).convert("RGB"))
    return out


def extract_frames(video_field: Any, num_frames: int) -> List[Image.Image]:
    """Return exactly `num_frames` PIL RGB frames sampled uniformly from a clip.

    Accepts: a path, a {'path','bytes'} dict (datasets Video(decode=False)),
    or an already-decoded decoder object. Tries decord first, then OpenCV.
    """
    if not isinstance(video_field, (str, dict)):
        return _frames_from_decoder(video_field, num_frames)

    path, tmp = _resolve_video_path(video_field)
    try:
        try:
            frames = _decode_with_decord(path, num_frames)
        except Exception:
            frames = _decode_with_opencv(path, num_frames)
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)

    while len(frames) < num_frames:  # safety pad
        frames.append(frames[-1])
    return frames[:num_frames]


def prep_audio(arr, max_seconds: Optional[float] = None, target_sr: int = TARGET_SR) -> np.ndarray:
    """Down-mix to mono float32 and optionally truncate to `max_seconds`."""
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim > 1:
        a = a.mean(axis=int(np.argmin(a.shape)))  # average channel dim
    a = a.reshape(-1)
    if max_seconds:
        a = a[: int(max_seconds * target_sr)]
    if a.size == 0:
        a = np.zeros(target_sr, dtype=np.float32)
    return a


def extract_audio(path: str, target_sr: int = TARGET_SR) -> np.ndarray:
    """Load a waveform (16 kHz mono float32) from a .wav OR a video file.

    Used at inference time when a separate audio file is not provided.
    """
    # decord can pull the audio track straight out of a video container
    try:
        from decord import AudioReader, cpu

        ar = AudioReader(path, ctx=cpu(0), sample_rate=target_sr, mono=True)
        a = ar[:].asnumpy().reshape(-1).astype(np.float32)
        if a.size:
            return a
    except Exception:
        pass
    # torchaudio handles .wav directly and most video containers via ffmpeg
    try:
        import torchaudio

        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        return wav.reshape(-1).numpy().astype(np.float32)
    except Exception as exc:
        raise RuntimeError(f"Could not load audio from {path}: {exc}")


# ---------------------------------------------------------------------------
# torch Dataset + collator
# ---------------------------------------------------------------------------
class RavdessAVDataset(torch.utils.data.Dataset):
    def __init__(self, hf_ds, num_frames: int, max_audio_seconds: float = 6.0):
        self.ds = hf_ds
        self.num_frames = num_frames
        self.max_audio_seconds = max_audio_seconds

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int):
        ex = self.ds[i]
        frames = extract_frames(ex["video"], self.num_frames)
        audio = prep_audio(load_audio(ex["audio"]), self.max_audio_seconds)
        return {"frames": frames, "audio": audio, "label": int(ex["emotion"])}


class AVCollator:
    """Turn raw frames/waveforms into batched model-ready tensors."""

    def __init__(self, image_processor, feature_extractor, num_frames: int):
        self.image_processor = image_processor
        self.feature_extractor = feature_extractor
        self.num_frames = num_frames

    def __call__(self, batch):
        all_frames, audios, labels = [], [], []
        for ex in batch:
            all_frames.extend(ex["frames"])  # flatten B*N frames
            audios.append(ex["audio"])
            labels.append(ex["label"])

        pixel_values = self.image_processor(images=all_frames, return_tensors="pt")["pixel_values"]

        fe = self.feature_extractor(
            audios,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        attn = fe.get("attention_mask")
        if attn is not None:
            input_lengths = attn.sum(-1).long()
        else:
            input_lengths = torch.tensor([len(a) for a in audios], dtype=torch.long)

        return {
            "pixel_values": pixel_values,                      # (B*N, 3, H, W)
            "n_frames": self.num_frames,                       # int
            "input_values": fe["input_values"],                # (B, T)
            "input_lengths": input_lengths,                    # (B,)
            "labels": torch.tensor(labels, dtype=torch.long),  # (B,)
        }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def _masked_mean(audio_model, hidden: torch.Tensor, input_lengths: Optional[torch.Tensor]) -> torch.Tensor:
    """Mean over the time axis, ignoring padded frames when lengths are known."""
    if input_lengths is None:
        return hidden.mean(dim=1)
    try:
        out_lens = audio_model._get_feat_extract_output_lengths(input_lengths)
        out_lens = torch.as_tensor(out_lens, device=hidden.device).long().clamp(1, hidden.size(1))
        rng = torch.arange(hidden.size(1), device=hidden.device)[None, :]
        mask = (rng < out_lens[:, None]).unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
    except Exception:
        return hidden.mean(dim=1)


class AVEmotionModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        siglip_id: str = DEFAULT_SIGLIP,
        wav2vec2_id: str = DEFAULT_WAV2VEC2,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        freeze: bool = True,
    ):
        super().__init__()
        self.freeze = freeze
        self.video_model = AutoModel.from_pretrained(siglip_id)
        self.audio_model = Wav2Vec2Model.from_pretrained(wav2vec2_id)

        vcfg = getattr(self.video_model.config, "vision_config", self.video_model.config)
        self.video_dim = getattr(vcfg, "hidden_size", None) or self.video_model.config.hidden_size
        self.audio_dim = self.audio_model.config.hidden_size

        #######################################################
        # Implement your classifier here
        #######################################################

        self.classifier = 

        if freeze:
            for p in self.video_model.parameters():
                p.requires_grad_(False)
            for p in self.audio_model.parameters():
                p.requires_grad_(False)
            self.video_model.eval()
            self.audio_model.eval()

    def train(self, mode: bool = True):
        # Keep frozen backbones in eval() (disable their dropout) regardless of mode.
        super().train(mode)
        if self.freeze:
            self.video_model.eval()
            self.audio_model.eval()
        return self

    def encode_video(self, pixel_values: torch.Tensor, n_frames: int) -> torch.Tensor:
        ctx = torch.no_grad() if self.freeze else nullcontext()
        with ctx:
            out = self.video_model.get_image_features(pixel_values=pixel_values)  # (B*N, Dv)
            feats = out.pooler_output if hasattr(out, "pooler_output") else out  # (B*N, Dv)
        return feats.view(-1, n_frames, feats.size(-1)).mean(dim=1)  # temporal avg pool -> (B, Dv)

    def encode_audio(self, input_values: torch.Tensor, input_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        ctx = torch.no_grad() if self.freeze else nullcontext()
        with ctx:
            hidden = self.audio_model(input_values=input_values).last_hidden_state  # (B, T, Da)
            return _masked_mean(self.audio_model, hidden, input_lengths)            # (B, Da)

    def forward(self, pixel_values, n_frames, input_values, input_lengths=None, labels=None):
        v = self.encode_video(pixel_values, n_frames)
        a = self.encode_audio(input_values, input_lengths)

        ###########################################################################
        # Concatenate the encodings, run the classifier and return the results.
        ###########################################################################


    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


def load_processors(siglip_id: str = DEFAULT_SIGLIP, wav2vec2_id: str = DEFAULT_WAV2VEC2):
    image_processor = AutoImageProcessor.from_pretrained(siglip_id)
    feature_extractor = AutoFeatureExtractor.from_pretrained(wav2vec2_id)
    return image_processor, feature_extractor


# ---------------------------------------------------------------------------
# Train / val split (no official split exists for RAVDESS_AV)
# ---------------------------------------------------------------------------
def _actor_from_video(video_field: Any) -> Optional[int]:
    path = None
    if isinstance(video_field, dict):
        path = video_field.get("path")
    elif isinstance(video_field, str):
        path = video_field
    if not path:
        return None
    m = _RAVDESS_RE.search(os.path.basename(path))
    return int(m.group(7)) if m else None


def make_split(ds, val_ratio: float = 0.2, seed: int = 42, subject_independent: bool = False):
    """Return (train_indices, val_indices, is_subject_independent).

    subject_independent=True keeps all clips of an actor on the same side of the
    split (the standard, harder RAVDESS protocol) when actor ids are recoverable
    from the video filenames; otherwise it falls back to a stratified split.
    """
    n = len(ds)
    labels = ds["emotion"]

    if subject_independent:
        actors = [_actor_from_video(v) for v in ds["video"]]
        if all(a is not None for a in actors) and len(set(actors)) > 1:
            uniq = sorted(set(actors))
            rng = random.Random(seed)
            rng.shuffle(uniq)
            n_val = max(1, int(round(len(uniq) * val_ratio)))
            val_actors = set(uniq[:n_val])
            train_idx = [i for i, a in enumerate(actors) if a not in val_actors]
            tmp_idx = [i for i, a in enumerate(actors) if a in val_actors]
            mid = len(tmp_idx) // 2
            val_idx=tmp_idx[:mid]
            tst_idx=tmp_idx[mid:]

            print('train idx')
            print(train_idx)
            print('val_idx')
            print(val_idx)
            print('tst_idx')
            print(tst_idx)

            return train_idx, val_idx, tst_idx, True
        print("[warn] could not recover actor ids -> falling back to stratified split")

    try:
        from sklearn.model_selection import train_test_split

        train_idx, val_idx = train_test_split(
            list(range(n)), test_size=val_ratio, random_state=seed, stratify=labels
        )
    except Exception:
        rng = random.Random(seed)
        idx = list(range(n))
        rng.shuffle(idx)
        k = int(n * (1 - val_ratio))
        train_idx, val_idx = idx[:k], idx[k:]
    return train_idx, val_idx, False
