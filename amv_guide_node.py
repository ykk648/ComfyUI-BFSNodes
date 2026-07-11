"""AMV Guide Builder — make the keyframes + blank guide video for the AMV av2av LoRA.

Modes (where the keyframe IMAGES come from):
  - auto_scene:      input video -> detect scene cuts -> keyframe at each cut
  - manual_indices:  input video + indices "0,16,34" -> those frames as keyframes
  - external_images: a batch of images (chain Load Image -> Image Batch) -> placed on the timeline

Placement (WHERE keyframes go, for external_images):
  - audio_beats: detect beats/onsets in the input `audio` and drop one image per beat (in order,
                 cycling the images if there are more beats than images). This syncs the cuts to
                 the music — exactly how AMVs are edited.
  - even:        evenly spaced across `length`
  - indices:     explicit `indices` string

Output is an IMAGE batch [length,H,W,C] (float 0..1), ready for VAE-encode / reference conditioning.
`hold` repeats each keyframe N frames so it survives the LTX 8x temporal VAE.
"""
import numpy as np
import torch

CATEGORY = "BFS/video"


def _parse_indices(s):
    out = []
    for tok in str(s).replace(";", ",").split(","):
        tok = tok.strip()
        if tok.isdigit():
            out.append(int(tok))
    return out


def _auto_cuts(video, threshold):
    """Content-diff scene cuts on [N,H,W,C]. Returns sorted cut indices (incl 0)."""
    cuts = [0]
    prev = video[0]
    for i in range(1, video.shape[0]):
        if (video[i] - prev).abs().mean().item() > threshold:
            cuts.append(i)
        prev = video[i]
    return cuts


def _audio_to_mono(audio):
    wf = audio["waveform"]            # [B, C, samples]
    sr = int(audio["sample_rate"])
    y = wf[0].mean(0).detach().cpu().numpy().astype(np.float32)
    return y, sr


def _beat_indices(audio, fps, length):
    """Beat (or onset) frame indices for the guide timeline. librosa if available, else RMS-onset."""
    y, sr = _audio_to_mono(audio)
    times = None
    try:
        import librosa
        _, beats = librosa.beat.beat_track(y=y, sr=sr)
        times = librosa.frames_to_time(beats, sr=sr)
        if len(times) < 2:  # fall back to onsets for sparse beats
            on = librosa.onset.onset_detect(y=y, sr=sr, units="time")
            if len(on) > len(times):
                times = on
    except Exception:
        # fallback: RMS energy peaks
        hop = max(1, sr // 100)
        rms = np.array([np.sqrt(np.mean(y[i:i + hop] ** 2)) for i in range(0, len(y), hop)])
        d = np.diff(rms, prepend=rms[:1])
        thr = d.mean() + d.std()
        peaks = np.where(d > thr)[0]
        times = peaks * hop / sr
    idx = sorted({int(round(t * fps)) for t in times if 0 <= t * fps < length})
    if 0 not in idx:
        idx = [0] + idx
    return idx


class AmvGuideBuilder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["auto_scene", "manual_indices", "external_images"],),
                "length": ("INT", {"default": 121, "min": 1, "max": 100000}),
                "hold": ("INT", {"default": 4, "min": 1, "max": 64,
                                 "tooltip": "Frames each keyframe is shown (>=4 survives LTX 8x temporal VAE)."}),
                "fill": (["white", "black"],),
                "placement": (["audio_beats", "even", "indices"],
                              {"tooltip": "external_images: how to position the images on the timeline."}),
                "fps": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 120.0,
                                  "tooltip": "Used to convert audio beat times -> frame indices."}),
            },
            "optional": {
                "video": ("IMAGE", {"tooltip": "Source frames (auto_scene / manual_indices)."}),
                "images": ("IMAGE", {"tooltip": "Keyframe images batch (external_images)."}),
                "audio": ("AUDIO", {"tooltip": "Drives placement=audio_beats (sync cuts to the music)."}),
                "indices": ("STRING", {"default": "", "tooltip": "Comma list e.g. '0,16,34' (placement=indices / manual)."}),
                "scene_threshold": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("guide", "debug")
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Builds the AMV guide (keyframes + blank). external_images can auto-place images "
                   "on the music's beats (placement=audio_beats). Feed `guide` into VAE/reference.")

    def build(self, mode, length, hold, fill, placement, fps,
              video=None, images=None, audio=None, indices="", scene_threshold=0.10):
        idx = _parse_indices(indices)
        keyframes = []  # (frame_index, image[H,W,C])

        if mode == "external_images":
            if images is None or images.shape[0] == 0:
                raise ValueError("external_images mode needs an `images` batch.")
            H, W, n = images.shape[1], images.shape[2], images.shape[0]
            # decide placement positions
            if placement == "audio_beats":
                if audio is None:
                    raise ValueError("placement=audio_beats needs an `audio` input.")
                pos = _beat_indices(audio, fps, length)
            elif placement == "indices" and idx:
                pos = idx
            else:  # even
                pos = [round(k * (length - 1) / max(1, n - 1)) for k in range(n)] if n > 1 else [0]
            # map one image per position (cycle images if fewer than positions)
            for k, at in enumerate(pos):
                keyframes.append((min(int(at), length - 1), images[k % n]))
        else:
            if video is None or video.shape[0] == 0:
                raise ValueError(f"{mode} mode needs a `video`.")
            H, W = video.shape[1], video.shape[2]
            cuts = _auto_cuts(video, scene_threshold) if mode == "auto_scene" else (idx or [0])
            for ci in cuts:
                if 0 <= ci < video.shape[0]:
                    keyframes.append((min(ci, length - 1), video[ci]))

        val = 1.0 if fill == "white" else 0.0
        guide = torch.full((length, H, W, 3), val, dtype=torch.float32)
        painted = []
        for at, frame in keyframes:
            f = frame[..., :3].to(torch.float32)
            for h in range(hold):
                if at + h < length:
                    guide[at + h] = f
            painted.append(at)

        dbg = (f"AMV guide | mode={mode} placement={placement if mode=='external_images' else '-'} | "
               f"{W}x{H} {length}f hold={hold} fill={fill} | {len(keyframes)} keyframes @ {sorted(set(painted))}")
        return (guide, dbg)


NODE_CLASS_MAPPINGS = {"BFSAmvGuideBuilder": AmvGuideBuilder}
NODE_DISPLAY_NAME_MAPPINGS = {"BFSAmvGuideBuilder": "AMV Guide Builder"}
