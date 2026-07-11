"""LTX-2.3 Identity Transfer — MULTIPLE ANGLES (temporary tester).

A copy of "LTX Identity Transfer" (overlap + source_phase, ltx_identity_overlap.py) that accepts
several reference views, each injected as its OWN overlap segment tagged with its OWN source_id —
exactly the multi-view training convention:

    reference_face_front   -> source_id 2   (REQUIRED; also drives the ArcFace projector)
    reference_back_head    -> source_id 3   (optional — back of the head / nape, hair)
    reference_body_front   -> source_id 4   (optional — full-body front, build/proportions)
    reference_side_profile -> source_id 5   (optional — side / profile of the face)

Empty optional inputs are simply omitted — matching the training view-dropout, so the model
degrades gracefully when you only provide the face. Non-frontal views feed only the overlap
latent (no ArcFace); the frontal face additionally drives the IdentityProjector tokens.

Load the matching multi-view LoRA on MODEL first. Reuses the exact ArcFace/projector/patch
machinery from ltx_identity_overlap; only the model patch is generalized to N reference blocks.
"""
import logging
import os
import types

import torch

from .ltx_identity_overlap import (
    _arcface_embed,
    _append_ctx_tokens,
    _find_ltxv,
    _load_projector,
)

log = logging.getLogger("LTXIdentityMultiAngle")

# Fixed viewpoint -> source_id convention (must match the multi-view training config).
_SLOT_SOURCE_ID = {"face": 2.0, "back": 3.0, "body": 4.0, "side": 5.0}
_DEBUG = os.environ.get("LTX_IDOVERLAP_DEBUG", "0") == "1"


def _dbg(*a):
    if _DEBUG:
        print("[LTXIdMulti] " + " ".join(str(x) for x in a), flush=True)


def _rotate_block(pe, start, length, seg, theta=10000.0):
    """Rotate RoPE freqs of tokens [start:start+length] by phase = seg * theta^(-d/L).
    Same math as ltx_identity_overlap._rotate_ref_freqs, but on an arbitrary token slice
    (each reference block sits at its own offset after the target)."""
    if length <= 0 or seg == 0.0:
        return pe
    cos, sin = pe[0], pe[1]
    rest = tuple(pe[2:])
    L = cos.shape[-1]
    d = torch.arange(L, device=cos.device, dtype=torch.float32)
    rate = theta ** (-d / float(L))
    phase = seg * rate
    pc = phase.cos().to(cos.dtype)
    ps = phase.sin().to(sin.dtype)
    idx = [slice(None)] * cos.dim()
    idx[-2] = slice(start, start + length)
    idx = tuple(idx)
    c0, s0 = cos[idx], sin[idx]
    cos = cos.clone(); sin = sin.clone()
    cos[idx] = c0 * pc - s0 * ps
    sin[idx] = s0 * pc + c0 * ps
    _dbg("rotate block start", start, "len", length, "seg", seg)
    return (cos, sin, *rest)


def _install_multi_patches(ltxv):
    """Idempotent multi-reference version of the overlap patch. Appends a LIST of reference
    latents (each with its own source_phase segment) after the target tokens, and trims them
    all before unpatchify. Uses distinct attribute/flag names so it never collides with the
    single-ref LTXIdentityTransfer patch."""
    if getattr(ltxv, "_id_multi_patched", False):
        return
    orig_process_input = ltxv._process_input
    orig_prepare_ts = ltxv._prepare_timestep
    orig_prepare_pe = ltxv._prepare_positional_embeddings
    orig_process_output = ltxv._process_output

    def process_input(self, x, keyframe_idxs, denoise_mask, **kw):
        # Reset per-forward state FIRST so stale values from a previous run can never leak
        # into this forward (e.g. if the ref latents stop reaching us after a Comfy update).
        self._idma_ref_len = 0
        self._idma_blocks = []
        self._idma_target_len = None
        self._idma_ppf = None
        out = orig_process_input(x, keyframe_idxs, denoise_mask, **kw)
        # Comfy versions differ on whether custom transformer_options keys arrive flattened
        # in kwargs or nested under kwargs["transformer_options"] — accept both.
        ref_lats = kw.get("_id_ref_latents")
        if ref_lats is None:
            ref_lats = (kw.get("transformer_options") or {}).get("_id_ref_latents")
        if not ref_lats:
            _dbg("process_input: NO ref latents in kwargs (flat or nested) -> vanilla forward. kw keys:", list(kw.keys()))
            return out
        from comfy.ldm.lightricks.model import latent_to_pixel_coords
        xx, pix, add = out
        is_av = isinstance(xx, (list, tuple))
        vx = xx[0] if is_av else xx
        vco = pix[0] if is_av else pix
        target_len = vx.shape[1]
        self._idma_target_len = target_len
        segs = getattr(self, "_idma_segs", [2.0] * len(ref_lats))
        blocks = []
        offset = target_len
        for ref_lat, seg in zip(ref_lats, segs):
            rt, rlc = self.patchifier.patchify(ref_lat.to(dtype=vx.dtype, device=vx.device))
            rpc = latent_to_pixel_coords(latent_coords=rlc, scale_factors=self.vae_scale_factors,
                                         causal_fix=self.causal_temporal_positioning)
            rt = self.patchify_proj(rt)
            if rt.shape[0] != vx.shape[0]:
                rt = rt.expand(vx.shape[0], -1, -1)
            if rpc.shape[0] != vco.shape[0]:
                rpc = rpc.expand(vco.shape[0], *([-1] * (rpc.dim() - 1)))
            rlen = rt.shape[1]
            vx = torch.cat([vx, rt], dim=1)
            vco = torch.cat([vco, rpc.to(vco)], dim=2)
            blocks.append((offset, rlen, float(seg)))
            offset += rlen
        self._idma_ref_len = offset - target_len
        self._idma_blocks = blocks
        # each ref is one latent frame at the target grid -> its token count == tokens per frame
        self._idma_ppf = blocks[0][1] if blocks else None
        add = dict(add); add["_id_ref_len"] = self._idma_ref_len
        _dbg("process_input OUT: blocks", blocks, "| target_len", target_len, "| total_ref", self._idma_ref_len)
        if is_av:
            xx = [vx, xx[1]]; pix = [vco, pix[1]]
        else:
            xx, pix = vx, vco
        return xx, pix, add

    def prepare_timestep(self, timestep, batch_size, hidden_dtype, **kw):
        # MEASURE-based extension: only pad the timestep by exactly what process_input
        # actually appended in THIS forward, and detect the timestep granularity from its
        # real length (per-token / per-frame / already-extended) instead of assuming one.
        # This keeps modulation and vx in lockstep across ComfyUI versions.
        ref_len = getattr(self, "_idma_ref_len", 0)
        tgt_len = getattr(self, "_idma_target_len", None)
        if ref_len and tgt_len:
            ppf = getattr(self, "_idma_ppf", None)
            if timestep.dim() <= 1:
                timestep = timestep.reshape(-1, 1).expand(batch_size, tgt_len).contiguous()
            cur = timestep.shape[1]
            # With audio connected, LTXAV may produce a combined video+audio per-token
            # timestep (e.g. 24576 = video 6144 + audio 18432). Trim to video-only so
            # the extension matches vx (which only has video + ref tokens).
            if cur > tgt_len + ref_len:
                _dbg("prepare_timestep: oversized", cur, "(tgt", tgt_len, "+ ref", ref_len,
                     ") -> trimming to video-only", tgt_len)
                timestep = timestep[:, :tgt_len]
                cur = tgt_len
            if cur == tgt_len:            # per-token, target only -> append per-token zeros
                z = torch.zeros(timestep.shape[0], ref_len, *timestep.shape[2:],
                                device=timestep.device, dtype=timestep.dtype)
                timestep = torch.cat([timestep, z], dim=1)
                _dbg("prepare_timestep: per-token extend", cur, "->", timestep.shape[1])
            elif ppf and cur * ppf == tgt_len:   # per-FRAME timestep -> append ref frames
                n_ref_frames = max(1, ref_len // ppf)
                z = torch.zeros(timestep.shape[0], n_ref_frames, *timestep.shape[2:],
                                device=timestep.device, dtype=timestep.dtype)
                timestep = torch.cat([timestep, z], dim=1)
                _dbg("prepare_timestep: per-frame extend", cur, "->", timestep.shape[1])
            elif cur == tgt_len + ref_len:       # already extended (double-patch guard)
                _dbg("prepare_timestep: already extended", cur)
            else:
                _dbg("prepare_timestep: UNEXPECTED len", cur, "target", tgt_len, "ppf", ppf, "-> untouched")
            # Guide nodes (IC-LoRA Guide / Director Guide, etc.) inject a grid_mask that
            # FILTERS x tokens in _process_input (x = x[:, grid_mask]) and later indexes the
            # timestep (timestep[:, grid_mask]). Our ref tokens are appended AFTER that
            # filter, so extend the mask with True for them (False would drop our clean
            # timesteps from the modulation and desync it from vx). Pad size is measured
            # from the actual gap so per-token AND per-frame masks both work.
            gm = kw.get("grid_mask")
            if gm is not None and hasattr(gm, "shape") and timestep.dim() >= 2:
                gap = timestep.shape[1] - gm.shape[-1]
                if 0 < gap <= ref_len:
                    pad = torch.ones(*gm.shape[:-1], gap, dtype=gm.dtype, device=gm.device)
                    kw = dict(kw); kw["grid_mask"] = torch.cat([gm, pad], dim=-1)
                    _dbg("prepare_timestep: grid_mask extended by", gap)
        return orig_prepare_ts(timestep, batch_size, hidden_dtype, **kw)

    def prepare_pe(self, pixel_coords, frame_rate, x_dtype):
        pe = orig_prepare_pe(pixel_coords, frame_rate, x_dtype)
        blocks = getattr(self, "_idma_blocks", [])
        if not blocks:
            return pe
        theta = getattr(self, "_id_rope_theta", 10000.0)

        def rot(v_pe):
            for (start, length, seg) in blocks:
                v_pe = _rotate_block(v_pe, start, length, seg, theta)
            return v_pe

        if isinstance(pe, list) and len(pe) and isinstance(pe[0], (list, tuple)) and isinstance(pe[0][0], (list, tuple)):
            v_pe, cross_v = pe[0][0], pe[0][1]
            v_pe = rot(v_pe)
            pe = [(v_pe, cross_v), pe[1]]
        else:
            pe = rot(pe)
        return pe

    def process_output(self, x, embedded_timestep, keyframe_idxs, **kw):
        ref_len = getattr(self, "_idma_ref_len", 0)
        if ref_len:
            _dbg("process_output: trimming", ref_len, "ref tokens")
        if ref_len:
            from comfy.ldm.lightricks.av_model import CompressedTimestep
            import copy
            if isinstance(x, (list, tuple)):
                x = [x[0][:, :x[0].shape[1] - ref_len], *x[1:]]
                et_list = list(embedded_timestep) if isinstance(embedded_timestep, (list, tuple)) else [embedded_timestep]
                v_et = et_list[0]
                if isinstance(v_et, CompressedTimestep):
                    ppf = max(1, getattr(v_et, "patches_per_frame", 1) or 1)
                    n_ref_frames = max(1, ref_len // ppf)
                    v_et2 = copy.copy(v_et)
                    v_et2.data = v_et.data[:, : v_et.num_frames - n_ref_frames].contiguous()
                    v_et2.num_frames = v_et.num_frames - n_ref_frames
                    et_list[0] = v_et2
                elif hasattr(v_et, "shape") and v_et.dim() >= 2 and v_et.shape[1] > 1:
                    et_list[0] = v_et[:, : v_et.shape[1] - ref_len]
                embedded_timestep = et_list
            else:
                x = x[:, :x.shape[1] - ref_len]
                if hasattr(embedded_timestep, "shape") and embedded_timestep.dim() >= 2 and embedded_timestep.shape[1] > 1:
                    embedded_timestep = embedded_timestep[:, : embedded_timestep.shape[1] - ref_len]
        return orig_process_output(x, embedded_timestep, keyframe_idxs, **kw)

    ltxv._process_input = types.MethodType(process_input, ltxv)
    ltxv._prepare_timestep = types.MethodType(prepare_timestep, ltxv)
    ltxv._prepare_positional_embeddings = types.MethodType(prepare_pe, ltxv)
    ltxv._process_output = types.MethodType(process_output, ltxv)
    ltxv._id_multi_patched = True
    log.info("LTXIdentityMultiAngle patches installed on %s", type(ltxv).__name__)


class LTXIdentityMultiAngle:
    @classmethod
    def INPUT_TYPES(cls):
        try:
            import folder_paths
            proj_choices = ["None"] + folder_paths.get_filename_list("loras")
        except Exception:
            proj_choices = ["None"]
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "latent": ("LATENT",),
                "reference_face_front": ("IMAGE", {
                    "tooltip": "REQUIRED — frontal face (source_id 2). Also drives the ArcFace projector."}),
                "identity_projector": (proj_choices, {"default": "None",
                    "tooltip": "ArcFace projector .safetensors (from models/loras). 'None' = overlap only."}),
                "phase_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.1,
                    "tooltip": "Global RoPE phase scale (training used 1.0). Applied to every view's source_id."}),
                "id_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 50.0, "step": 0.5,
                    "tooltip": "Multiplies the ArcFace projector tokens (frontal face only)."}),
                "arcface_mode": (["auto_adjust", "as_is", "disable"], {"default": "auto_adjust"}),
                "debug_log": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "reference_back_head": ("IMAGE", {
                    "tooltip": "OPTIONAL — back of the head / nape, hair from behind (source_id 3). Leave empty to skip."}),
                "reference_body_front": ("IMAGE", {
                    "tooltip": "OPTIONAL — full-body front, build/proportions (source_id 4). Leave empty to skip."}),
                "reference_side_profile": ("IMAGE", {
                    "tooltip": "OPTIONAL — side / profile of the face (source_id 5). Leave empty to skip."}),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("model", "positive", "negative", "latent", "debug")
    FUNCTION = "apply"
    CATEGORY = "LTX/identity"
    DESCRIPTION = ("Multi-angle identity transfer: frontal face (required) + optional back/body/side "
                   "reference views, each injected as its own overlap+source_phase segment (source_id "
                   "2/3/4/5), matching the multi-view training. Load the multi-view LoRA on MODEL first.")

    def apply(self, model, positive, negative, vae, latent, reference_face_front,
              identity_projector="None", phase_scale=1.0, id_strength=1.0,
              arcface_mode="auto_adjust", debug_log=False,
              reference_back_head=None, reference_body_front=None, reference_side_profile=None):
        import comfy.utils

        global _DEBUG
        _DEBUG = bool(debug_log)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m = model.clone()
        ltxv = _find_ltxv(m)

        _, w_sf, h_sf = vae.downscale_index_formula
        _, _, _, lat_h, lat_w = latent["samples"].shape
        px_w, px_h = lat_w * w_sf, lat_h * h_sf

        def encode(img):
            ref_px = comfy.utils.common_upscale(img.movedim(-1, 1), px_w, px_h, "bilinear", "center").movedim(1, -1)[:1, :, :, :3]
            return vae.encode(ref_px)

        # Build the provided views in a fixed order (face first). Empty optionals are skipped.
        provided = [("face", reference_face_front)]
        if reference_back_head is not None:
            provided.append(("back", reference_back_head))
        if reference_body_front is not None:
            provided.append(("body", reference_body_front))
        if reference_side_profile is not None:
            provided.append(("side", reference_side_profile))

        ref_lats, segs, used = [], [], []
        for slot, img in provided:
            ref_lats.append(encode(img))
            segs.append(_SLOT_SOURCE_ID[slot] * float(phase_scale))
            used.append(f"{slot}(sid={_SLOT_SOURCE_ID[slot]:.0f})")

        _install_multi_patches(ltxv)
        ltxv._idma_segs = segs
        ltxv._id_rope_theta = 10000.0
        m.model_options = dict(m.model_options)
        to = dict(m.model_options.get("transformer_options", {}))
        to["_id_ref_latents"] = ref_lats
        m.model_options["transformer_options"] = to

        # ArcFace projector tokens from the FRONTAL face only (optional).
        use_projector = identity_projector not in (None, "", "None")
        emb = _arcface_embed(reference_face_front, mode=arcface_mode) if use_projector else None
        if not use_projector:
            arc_status = "no projector (overlap only)"
        elif arcface_mode == "disable":
            arc_status = "disabled"
        else:
            arc_status = "OK" if emb is not None else "NO FACE -> skipped"
        if emb is not None:
            path = identity_projector
            if not os.path.isabs(path):
                try:
                    import folder_paths
                    resolved = folder_paths.get_full_path("loras", identity_projector)
                    if resolved:
                        path = resolved
                except Exception:
                    pass
            if not os.path.isabs(path) or not os.path.exists(path):
                for b in ("models/loras", "models", "."):
                    if os.path.exists(os.path.join(b, identity_projector)):
                        path = os.path.join(b, identity_projector); break
            projector = _load_projector(path, device)
            emb = emb.to(device=device, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                id_tok = projector(emb) * float(id_strength)
                unc = projector(torch.zeros(1, projector.in_dim, device=device))
            positive = _append_ctx_tokens(positive, id_tok)
            negative = _append_ctx_tokens(negative, unc)

        dbg = (
            "=== LTX Identity Transfer — MULTIPLE ANGLES ===\n"
            f"reference views ({len(ref_lats)}): {', '.join(used)}\n"
            f"ref latents: {[list(r.shape) for r in ref_lats]} | phase_scale={phase_scale}\n"
            f"arcface (frontal): {arc_status} (mode={arcface_mode}) | id_strength={id_strength}\n"
            f"patches on {type(ltxv).__name__}: process_input/prepare_timestep/prepare_pe/process_output (multi-block)\n"
            "Empty optional inputs are omitted (matches training view-dropout). Connect negative + CFG 3-5, no LightX2V."
        )
        log.info("\n" + dbg)
        return (m, positive, negative, latent, dbg)


NODE_CLASS_MAPPINGS = {"LTXIdentityMultiAngle": LTXIdentityMultiAngle}
NODE_DISPLAY_NAME_MAPPINGS = {"LTXIdentityMultiAngle": "LTX Identity Transfer (Multiple Angles)"}
