"""LTX Identity — Gemma-Vision Conditioning (MagicMirror-Caminho2).

The identity breakthrough for LTX: the model conditions in Gemma's embedding space, so foreign
encoders (CLIP/DINO/ArcFace adapters) fail. Instead we feed the REFERENCE IMAGE through LTX's OWN
multimodal Gemma 3 (its text encoder is Gemma 3 with a vision tower), so the reference becomes
conditioning tokens IN THE NATIVE SPACE — the model reads the person like text. No reference
latent (no copy-paste/mask), no foreign encoder, no DiT surgery. Pair with a LoRA trained the
same way (reference image -> Gemma-vision conditioning + ArcFace loss).

Everything comes from the CLIP already loaded in the workflow: ComfyUI's native LTX Gemma-3
tokenizer already accepts an image (clip.tokenize(prompt, image=...) inserts <image_soft_token>
and carries the pixels), and clip.encode_from_tokens runs the multimodal Gemma + projection +
connector. So this node is just: tokenize(prompt, image) -> encode -> CONDITIONING.
"""
from __future__ import annotations


class LTXIdentityGemmaVision:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "reference_image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "ref_t2v: a person in a room, medium shot."}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "BFS/LTX Identity"

    def encode(self, clip, reference_image, prompt):
        # The LTX Gemma-3 tokenizer takes the ComfyUI IMAGE directly (movedim + <image_soft_token>),
        # and encode runs the multimodal Gemma. Identity ends up in the native conditioning space.
        try:
            tokens = clip.tokenize(prompt, image=reference_image)
        except TypeError:
            # Older signatures may name it differently; fall back to positional image.
            tokens = clip.tokenize(prompt, reference_image)
        cond = clip.encode_from_tokens_scheduled(tokens)
        print("[BFS Gemma-Vision] multimodal identity CONDITIONING ready (reference in native space).")
        return (cond,)


NODE_CLASS_MAPPINGS = {"LTXIdentityGemmaVision": LTXIdentityGemmaVision}
NODE_DISPLAY_NAME_MAPPINGS = {"LTXIdentityGemmaVision": "LTX Identity Gemma-Vision (Caminho2)"}
