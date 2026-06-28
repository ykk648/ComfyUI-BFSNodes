import logging
import math
import os

import numpy as np
import torch
from PIL import Image

from .util import (
    tensor_to_pil,
    pil_to_tensor,
    fit_inside,
    aligned_offset,
    paste_with_alpha,
    add_white_padding,
)


log = logging.getLogger("BFS.Nodes")
_INSIGHTFACE_APP = None
_INSIGHTFACE_UNAVAILABLE = False


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _parse_rgb_color(value, default=(255, 255, 255)):
    try:
        parts = [int(x.strip()) for x in str(value).split(",")]
        if len(parts) != 3:
            return default
        return tuple(_clamp(p, 0, 255) for p in parts)
    except Exception:
        return default


def _select_face_box(boxes, selection, image_w, image_h):
    if not boxes:
        return None

    def area(item):
        x1, y1, x2, y2 = item["bbox"]
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    if selection == "leftmost":
        return min(boxes, key=lambda item: (item["bbox"][0] + item["bbox"][2]) * 0.5)
    if selection == "rightmost":
        return max(boxes, key=lambda item: (item["bbox"][0] + item["bbox"][2]) * 0.5)
    if selection == "center":
        cx0, cy0 = image_w * 0.5, image_h * 0.38

        def center_distance(item):
            x1, y1, x2, y2 = item["bbox"]
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            return (cx - cx0) ** 2 + (cy - cy0) ** 2

        return min(boxes, key=center_distance)

    return max(boxes, key=area)


def _get_insightface_app(det_size):
    global _INSIGHTFACE_APP, _INSIGHTFACE_UNAVAILABLE

    if _INSIGHTFACE_UNAVAILABLE:
        return None
    if _INSIGHTFACE_APP is not None:
        return _INSIGHTFACE_APP

    model_root = os.path.expanduser("~/.insightface")
    det_model = os.path.join(model_root, "models", "buffalo_l", "det_10g.onnx")
    if not os.path.exists(det_model):
        _INSIGHTFACE_UNAVAILABLE = True
        return None

    try:
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(name="buffalo_l", root=model_root, allowed_modules=["detection"])
        app.prepare(ctx_id=-1, det_size=(int(det_size), int(det_size)))
        _INSIGHTFACE_APP = app
        return app
    except Exception as exc:
        _INSIGHTFACE_UNAVAILABLE = True
        log.warning("InsightFace detector unavailable, falling back to OpenCV: %s", exc)
        return None


def _detect_faces_insightface(image_np, det_size, min_confidence):
    app = _get_insightface_app(det_size)
    if app is None:
        return []

    try:
        image_bgr = image_np[:, :, ::-1].copy()
        faces = app.get(image_bgr)
    except Exception as exc:
        log.warning("InsightFace detection failed, falling back to OpenCV: %s", exc)
        return []

    h, w = image_np.shape[:2]
    boxes = []
    for face in faces:
        score = float(getattr(face, "det_score", 1.0))
        if score < min_confidence:
            continue
        x1, y1, x2, y2 = [float(v) for v in face.bbox[:4]]
        x1 = _clamp(x1, 0.0, float(w - 1))
        y1 = _clamp(y1, 0.0, float(h - 1))
        x2 = _clamp(x2, x1 + 1.0, float(w))
        y2 = _clamp(y2, y1 + 1.0, float(h))
        boxes.append({"bbox": (x1, y1, x2, y2), "score": score, "source": "insightface"})
    return boxes


def _detect_faces_opencv(image_np, min_face_size_pct):
    try:
        import cv2
    except Exception:
        return []

    h, w = image_np.shape[:2]
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    min_size = max(12, int(min(w, h) * max(0.0, min_face_size_pct) / 100.0))
    cascade_names = [
        "haarcascade_frontalface_alt2.xml",
        "haarcascade_frontalface_default.xml",
        "haarcascade_profileface.xml",
    ]
    boxes = []

    for name in cascade_names:
        path = os.path.join(cv2.data.haarcascades, name)
        cascade = cv2.CascadeClassifier(path)
        if cascade.empty():
            continue

        detections = cascade.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=4,
            minSize=(min_size, min_size),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        for x, y, bw, bh in detections:
            boxes.append({
                "bbox": (float(x), float(y), float(x + bw), float(y + bh)),
                "score": 0.5,
                "source": f"opencv:{name}",
            })

        if name == "haarcascade_profileface.xml":
            flipped = cv2.flip(gray, 1)
            detections = cascade.detectMultiScale(
                flipped,
                scaleFactor=1.08,
                minNeighbors=4,
                minSize=(min_size, min_size),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
            for x, y, bw, bh in detections:
                boxes.append({
                    "bbox": (float(w - x - bw), float(y), float(w - x), float(y + bh)),
                    "score": 0.5,
                    "source": "opencv:profile_flipped",
                })

    return boxes


def _fallback_head_box(image_w, image_h):
    side = min(image_w, max(1, int(round(image_h * 0.55))))
    cx = image_w * 0.5
    cy = image_h * 0.28
    x1 = cx - side * 0.5
    y1 = cy - side * 0.38
    return (x1, y1, x1 + side, y1 + side)


def _fit_square_box_to_image(crop_box, image_w, image_h):
    left, top, right, bottom = [float(v) for v in crop_box]
    side = max(1.0, right - left, bottom - top)
    side = min(side, float(max(1, min(image_w, image_h))))

    cx = (left + right) * 0.5
    cy = (top + bottom) * 0.5
    left = cx - side * 0.5
    top = cy - side * 0.5
    right = left + side
    bottom = top + side

    if left < 0:
        right -= left
        left = 0.0
    if top < 0:
        bottom -= top
        top = 0.0
    if right > image_w:
        left -= right - image_w
        right = float(image_w)
    if bottom > image_h:
        top -= bottom - image_h
        bottom = float(image_h)

    left = _clamp(left, 0.0, float(image_w - side))
    top = _clamp(top, 0.0, float(image_h - side))
    return (left, top, left + side, top + side)


def _crop_square_with_padding(pil_img, crop_box, output_size, pad_color):
    left, top, right, bottom = [int(round(v)) for v in crop_box]
    side = max(1, max(right - left, bottom - top))
    right = left + side
    bottom = top + side

    src_left = _clamp(left, 0, pil_img.width)
    src_top = _clamp(top, 0, pil_img.height)
    src_right = _clamp(right, 0, pil_img.width)
    src_bottom = _clamp(bottom, 0, pil_img.height)

    canvas = Image.new("RGB", (side, side), pad_color)
    if src_right > src_left and src_bottom > src_top:
        crop = pil_img.crop((src_left, src_top, src_right, src_bottom))
        canvas.paste(crop, (src_left - left, src_top - top))

    if output_size > 0 and canvas.size != (output_size, output_size):
        canvas = canvas.resize((output_size, output_size), Image.LANCZOS)
    return canvas, (left, top, right, bottom)


# ---------------------------------------------------------------------------
# AutoCropHeadReference
# ---------------------------------------------------------------------------

class AutoCropHeadReference:
    """
    Detects a face in a portrait/body photo and returns a square head reference.

    Default behavior favors the local InsightFace detector when available, then
    falls back to OpenCV Haar cascades, then to an upper-body center crop.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "detector": (
                    ["auto", "insightface", "opencv_haar", "none"],
                    {"default": "auto"},
                ),
                "face_selection": (
                    ["largest", "center", "leftmost", "rightmost"],
                    {"default": "largest"},
                ),
                "output_size": (
                    "INT",
                    {"default": 768, "min": 128, "max": 2048, "step": 64},
                ),
                "side_padding_pct": (
                    "FLOAT",
                    {"default": 45.0, "min": 0.0, "max": 200.0, "step": 1.0},
                ),
                "top_padding_pct": (
                    "FLOAT",
                    {"default": 35.0, "min": 0.0, "max": 200.0, "step": 1.0},
                ),
                "bottom_padding_pct": (
                    "FLOAT",
                    {"default": 35.0, "min": 0.0, "max": 300.0, "step": 1.0},
                ),
                "vertical_shift_pct": (
                    "FLOAT",
                    {"default": -5.0, "min": -100.0, "max": 100.0, "step": 1.0},
                ),
                "min_confidence": (
                    "FLOAT",
                    {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "det_size": (
                    "INT",
                    {"default": 640, "min": 320, "max": 1280, "step": 32},
                ),
                "min_face_size_pct": (
                    "FLOAT",
                    {"default": 3.0, "min": 1.0, "max": 30.0, "step": 0.5},
                ),
                "pad_color": (
                    "STRING",
                    {"default": "255,255,255"},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("cropped_image", "debug")
    FUNCTION = "crop"
    CATEGORY = "BFS/image"

    def crop(
        self,
        image,
        detector="auto",
        face_selection="largest",
        output_size=768,
        side_padding_pct=45.0,
        top_padding_pct=35.0,
        bottom_padding_pct=35.0,
        vertical_shift_pct=-5.0,
        min_confidence=0.35,
        det_size=640,
        min_face_size_pct=3.0,
        pad_color="255,255,255",
    ):
        if len(image.shape) != 4 or image.shape[0] < 1:
            raise ValueError("Input 'image' must be a valid IMAGE batch.")

        output_size = int(output_size)
        pad_rgb = _parse_rgb_color(pad_color)
        out_images = []
        debug_lines = []

        for i in range(image.shape[0]):
            pil_img = tensor_to_pil(image[i]).convert("RGB")
            image_np = np.asarray(pil_img)
            h, w = image_np.shape[:2]

            boxes = []
            if detector in ("auto", "insightface"):
                boxes = _detect_faces_insightface(image_np, int(det_size), float(min_confidence))
            if not boxes and detector in ("auto", "opencv_haar"):
                boxes = _detect_faces_opencv(image_np, float(min_face_size_pct))

            selected = _select_face_box(boxes, face_selection, w, h)
            if selected is None or detector == "none":
                x1, y1, x2, y2 = _fallback_head_box(w, h)
                source = "fallback:upper_center"
                score = 0.0
            else:
                x1, y1, x2, y2 = selected["bbox"]
                source = selected["source"]
                score = selected["score"]

            face_w = max(1.0, x2 - x1)
            face_h = max(1.0, y2 - y1)
            rect_left = x1 - face_w * (float(side_padding_pct) / 100.0)
            rect_right = x2 + face_w * (float(side_padding_pct) / 100.0)
            rect_top = y1 - face_h * (float(top_padding_pct) / 100.0)
            rect_bottom = y2 + face_h * (float(bottom_padding_pct) / 100.0)

            side = max(rect_right - rect_left, rect_bottom - rect_top, 1.0)
            cx = (rect_left + rect_right) * 0.5
            cy = (rect_top + rect_bottom) * 0.5 + side * (float(vertical_shift_pct) / 100.0)
            crop_box = (
                cx - side * 0.5,
                cy - side * 0.5,
                cx + side * 0.5,
                cy + side * 0.5,
            )
            crop_box = _fit_square_box_to_image(crop_box, w, h)

            cropped, resolved_crop = _crop_square_with_padding(
                pil_img=pil_img,
                crop_box=crop_box,
                output_size=output_size,
                pad_color=pad_rgb,
            )
            out_images.append(pil_to_tensor(cropped))
            debug_lines.append(
                f"image[{i}] {w}x{h} detector={source} score={score:.3f} "
                f"face=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}) "
                f"crop={resolved_crop} output={cropped.width}x{cropped.height}"
            )

        return (torch.stack(out_images, dim=0), "\n".join(debug_lines))


# ---------------------------------------------------------------------------
# FrameRangedFaceLoader
# ---------------------------------------------------------------------------

class FrameRangedFaceLoader:
    """
    Wraps a single face IMAGE with a frame range [frame_start, frame_end].
    Returns a FACE_SEQUENCE — a list of dicts used by ReservedRegionFrameComposer.

    frame_end = -1 means "until the last frame" (no upper limit).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "frame_start": (
                    "INT",
                    {"default": 0, "min": 0, "max": 999999, "step": 1},
                ),
                "frame_end": (
                    "INT",
                    {"default": -1, "min": -1, "max": 999999, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("FACE_SEQUENCE",)
    RETURN_NAMES = ("face_sequence",)
    FUNCTION = "load"
    CATEGORY = "video/composition"

    def load(self, image, frame_start, frame_end):
        """
        image: IMAGE tensor [N, H, W, C] — only the first image is used.
        Returns a FACE_SEQUENCE list with a single entry.
        """
        face_pil = tensor_to_pil(image[0]).convert("RGBA")
        face_pil = add_white_padding(face_pil, 16)

        entry = {
            "image": face_pil,
            "frame_start": frame_start,
            "frame_end": frame_end,  # -1 == no upper limit
        }
        return ([entry],)


# ---------------------------------------------------------------------------
# FaceSequenceBatch
# ---------------------------------------------------------------------------

class FaceSequenceBatch:
    """
    Joins two FACE_SEQUENCE lists into one.
    Chain multiple nodes to build batches with many ranges.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "face_sequence_a": ("FACE_SEQUENCE",),
                "face_sequence_b": ("FACE_SEQUENCE",),
            }
        }

    RETURN_TYPES = ("FACE_SEQUENCE",)
    RETURN_NAMES = ("face_sequence",)
    FUNCTION = "batch"
    CATEGORY = "video/composition"

    def batch(self, face_sequence_a, face_sequence_b):
        return (face_sequence_a + face_sequence_b,)


# ---------------------------------------------------------------------------
# ReservedRegionFrameComposer
# ---------------------------------------------------------------------------

class ReservedRegionFrameComposer:
    """
    ComfyUI node that composes a reserved region (left/right/top/bottom)
    into every frame, while preserving the original output frame size.

    Main features:
    - Keeps final output resolution equal to original frame resolution
    - Fits video content into remaining area without crop
    - Fills reserved region with chroma color
    - Places one or many face images in that region
    - Supports temporal distribution modes for face batches (IMAGE input)
    - Supports per-frame ranges via FACE_SEQUENCE input
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),

                "region_position": (
                    ["left", "right", "top", "bottom"],
                    {"default": "left"}
                ),

                "region_size_px": (
                    "INT",
                    {"default": 320, "min": 8, "max": 8192, "step": 1}
                ),

                "face_distribution": (
                    [
                        "single_first",
                        "one_face_per_frame",
                        "one_face_per_interval",
                        "all_faces_every_frame"
                    ],
                    {"default": "one_face_per_interval"}
                ),

                "interval_frames": (
                    "INT",
                    {"default": 12, "min": 1, "max": 1000000, "step": 1}
                ),

                "overflow_mode": (
                    ["loop", "clamp", "error"],
                    {"default": "loop"}
                ),

                "stack_direction": (
                    ["auto", "vertical", "horizontal", "grid"],
                    {"default": "auto"}
                ),

                "face_scale_pct": (
                    "FLOAT",
                    {"default": 90.0, "min": 1.0, "max": 100.0, "step": 1.0}
                ),

                "face_padding_px": (
                    "INT",
                    {"default": 12, "min": 0, "max": 2048, "step": 1}
                ),

                "face_gap_px": (
                    "INT",
                    {"default": 12, "min": 0, "max": 2048, "step": 1}
                ),

                "face_align_main": (
                    ["start", "center", "end"],
                    {"default": "center"}
                ),

                "face_align_cross": (
                    ["start", "center", "end"],
                    {"default": "center"}
                ),

                "chroma_r": (
                    "INT",
                    {"default": 0, "min": 0, "max": 255, "step": 1}
                ),
                "chroma_g": (
                    "INT",
                    {"default": 255, "min": 0, "max": 255, "step": 1}
                ),
                "chroma_b": (
                    "INT",
                    {"default": 0, "min": 0, "max": 255, "step": 1}
                ),
            },
            "optional": {
                # When connected, overrides face_images completely.
                # face_distribution / interval_frames are ignored.
                "face_sequence": ("FACE_SEQUENCE",),

                # Legacy flat batch input (IMAGE). Used only when
                # face_sequence is NOT connected.
                "face_images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames_out",)
    FUNCTION = "process"
    CATEGORY = "video/composition"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_face(self, face_tensor):
        """Converts input tensor to RGBA PIL so alpha transparency is preserved."""
        face = tensor_to_pil(face_tensor).convert("RGBA")
        face = add_white_padding(face, 16)
        return face

    # --- Resolution for plain IMAGE input ---

    def _resolve_faces_for_frame_image(
        self,
        faces_pil,
        frame_idx,
        face_distribution,
        interval_frames,
        overflow_mode,
    ):
        """
        Returns the list of face PIL images for the current frame
        when a plain IMAGE batch is used (legacy mode).
        """
        total = len(faces_pil)

        if total == 0:
            raise ValueError("No face images were provided.")

        if face_distribution == "single_first":
            return [faces_pil[0]]

        if face_distribution == "one_face_per_frame":
            idx = frame_idx
            if idx < total:
                return [faces_pil[idx]]
            if overflow_mode == "loop":
                return [faces_pil[idx % total]]
            elif overflow_mode == "clamp":
                return [faces_pil[-1]]
            else:
                raise ValueError(
                    f"Face batch is too small for one_face_per_frame. "
                    f"Frame index {frame_idx} requires face index {idx}, "
                    f"but only {total} face images were provided."
                )

        if face_distribution == "one_face_per_interval":
            idx = frame_idx // interval_frames
            if idx < total:
                return [faces_pil[idx]]
            if overflow_mode == "loop":
                return [faces_pil[idx % total]]
            elif overflow_mode == "clamp":
                return [faces_pil[-1]]
            else:
                raise ValueError(
                    f"Face batch is too small for one_face_per_interval. "
                    f"Frame index {frame_idx} requires interval face index {idx}, "
                    f"but only {total} face images were provided."
                )

        if face_distribution == "all_faces_every_frame":
            return faces_pil

        raise ValueError(f"Unsupported face_distribution mode: {face_distribution}")

    # --- Resolution for FACE_SEQUENCE input ---

    def _resolve_faces_for_frame_sequence(self, face_sequence, frame_idx, overflow_mode):
        """
        Returns the list of face PIL images for the current frame
        when a FACE_SEQUENCE is used (range mode).

        Collects all entries whose [frame_start, frame_end] covers frame_idx.
        frame_end == -1 means no upper limit.

        If no entry matches:
          - loop: wraps around using sorted order by frame_start
          - clamp: uses the last entry (highest frame_start)
          - error: raises ValueError
        """
        matched = [
            e for e in face_sequence
            if e["frame_start"] <= frame_idx and (
                e["frame_end"] == -1 or frame_idx <= e["frame_end"]
            )
        ]

        if matched:
            return [e["image"] for e in matched]

        # Fallback
        sorted_seq = sorted(face_sequence, key=lambda e: e["frame_start"])
        if not sorted_seq:
            raise ValueError("FACE_SEQUENCE is empty.")

        if overflow_mode == "loop":
            total = len(sorted_seq)
            return [sorted_seq[frame_idx % total]["image"]]
        elif overflow_mode == "clamp":
            return [sorted_seq[-1]["image"]]
        else:
            raise ValueError(
                f"No face entry covers frame {frame_idx} and overflow_mode is 'error'."
            )

    # --- Layout helpers ---

    def _resize_single_face(self, face, region_w, region_h, face_scale_pct, face_padding_px):
        """Resizes one face to fit inside the usable region area."""
        usable_w = max(1, region_w - 2 * face_padding_px)
        usable_h = max(1, region_h - 2 * face_padding_px)

        target_w = max(1, int(round(usable_w * (face_scale_pct / 100.0))))
        target_h = max(1, int(round(usable_h * (face_scale_pct / 100.0))))

        fw, fh = face.size
        tw, th = fit_inside(fw, fh, target_w, target_h)
        return face.resize((tw, th), Image.LANCZOS)

    def _layout_faces_stack(
        self,
        faces_pil,
        region_w,
        region_h,
        face_scale_pct,
        face_padding_px,
        face_gap_px,
        stack_direction,
    ):
        """
        Prepares resized face images for stack mode.
        Returns a tuple describing the chosen layout and resized images.
        """
        usable_w = max(1, region_w - 2 * face_padding_px)
        usable_h = max(1, region_h - 2 * face_padding_px)

        count = len(faces_pil)
        if count == 0:
            return None

        if stack_direction == "auto":
            stack_direction = "vertical" if region_h >= region_w else "horizontal"

        items = []

        if stack_direction == "vertical":
            slot_h = max(1, (usable_h - face_gap_px * (count - 1)) // count)
            slot_w = usable_w

            for face in faces_pil:
                fw, fh = face.size
                tw, th = fit_inside(
                    fw, fh,
                    max(1, int(round(slot_w * (face_scale_pct / 100.0)))),
                    max(1, int(round(slot_h * (face_scale_pct / 100.0))))
                )
                items.append(face.resize((tw, th), Image.LANCZOS))

            return ("vertical", items, usable_w, usable_h)

        if stack_direction == "horizontal":
            slot_w = max(1, (usable_w - face_gap_px * (count - 1)) // count)
            slot_h = usable_h

            for face in faces_pil:
                fw, fh = face.size
                tw, th = fit_inside(
                    fw, fh,
                    max(1, int(round(slot_w * (face_scale_pct / 100.0)))),
                    max(1, int(round(slot_h * (face_scale_pct / 100.0))))
                )
                items.append(face.resize((tw, th), Image.LANCZOS))

            return ("horizontal", items, usable_w, usable_h)

        if stack_direction == "grid":
            cols = max(1, math.ceil(math.sqrt(count)))
            rows = max(1, math.ceil(count / cols))

            cell_w = max(1, (usable_w - face_gap_px * (cols - 1)) // cols)
            cell_h = max(1, (usable_h - face_gap_px * (rows - 1)) // rows)

            for face in faces_pil:
                fw, fh = face.size
                tw, th = fit_inside(
                    fw, fh,
                    max(1, int(round(cell_w * (face_scale_pct / 100.0)))),
                    max(1, int(round(cell_h * (face_scale_pct / 100.0))))
                )
                items.append(face.resize((tw, th), Image.LANCZOS))

            return ("grid", items, usable_w, usable_h, cols, rows)

        raise ValueError(f"Unsupported stack_direction: {stack_direction}")

    def _paste_single_face(
        self,
        canvas,
        face,
        region_x,
        region_y,
        region_w,
        region_h,
        region_position,
        face_padding_px,
        face_align_main,
        face_align_cross,
    ):
        """Pastes one face inside the reserved region."""
        tw, th = face.size
        area_w = max(1, region_w - 2 * face_padding_px)
        area_h = max(1, region_h - 2 * face_padding_px)

        if region_position in ["left", "right"]:
            local_x = face_padding_px + aligned_offset(area_w, tw, face_align_cross)
            local_y = face_padding_px + aligned_offset(area_h, th, face_align_main)
        else:
            local_x = face_padding_px + aligned_offset(area_w, tw, face_align_main)
            local_y = face_padding_px + aligned_offset(area_h, th, face_align_cross)

        paste_with_alpha(canvas, face, (region_x + local_x, region_y + local_y))

    def _paste_stack_faces(
        self,
        canvas,
        faces_pil,
        region_x,
        region_y,
        region_w,
        region_h,
        face_scale_pct,
        face_padding_px,
        face_gap_px,
        face_align_main,
        face_align_cross,
        stack_direction,
    ):
        """Pastes multiple faces inside the reserved region."""
        layout = self._layout_faces_stack(
            faces_pil=faces_pil,
            region_w=region_w,
            region_h=region_h,
            face_scale_pct=face_scale_pct,
            face_padding_px=face_padding_px,
            face_gap_px=face_gap_px,
            stack_direction=stack_direction,
        )

        if layout is None:
            return

        usable_x = region_x + face_padding_px
        usable_y = region_y + face_padding_px

        kind = layout[0]

        if kind == "vertical":
            items   = layout[1]
            usable_w = layout[2]
            usable_h = layout[3]
            total_h = sum(img.size[1] for img in items) + face_gap_px * (len(items) - 1)
            start_y = usable_y + aligned_offset(usable_h, total_h, face_align_main)

            y = start_y
            for img in items:
                x = usable_x + aligned_offset(usable_w, img.size[0], face_align_cross)
                paste_with_alpha(canvas, img, (x, y))
                y += img.size[1] + face_gap_px
            return

        if kind == "horizontal":
            items    = layout[1]
            usable_w = layout[2]
            usable_h = layout[3]
            total_w = sum(img.size[0] for img in items) + face_gap_px * (len(items) - 1)
            start_x = usable_x + aligned_offset(usable_w, total_w, face_align_main)

            x = start_x
            for img in items:
                y = usable_y + aligned_offset(usable_h, img.size[1], face_align_cross)
                paste_with_alpha(canvas, img, (x, y))
                x += img.size[0] + face_gap_px
            return

        if kind == "grid":
            items    = layout[1]
            usable_w = layout[2]
            usable_h = layout[3]
            cols     = layout[4]  # type: ignore[index]
            rows     = layout[5]  # type: ignore[index]

            cell_w = max(1, (usable_w - face_gap_px * (cols - 1)) // cols)
            cell_h = max(1, (usable_h - face_gap_px * (rows - 1)) // rows)

            grid_w = cols * cell_w + (cols - 1) * face_gap_px
            grid_h = rows * cell_h + (rows - 1) * face_gap_px

            start_x = usable_x + aligned_offset(usable_w, grid_w, face_align_main)
            start_y = usable_y + aligned_offset(usable_h, grid_h, face_align_cross)

            for idx, img in enumerate(items):
                row = idx // cols
                col = idx % cols

                cell_x = start_x + col * (cell_w + face_gap_px)
                cell_y = start_y + row * (cell_h + face_gap_px)

                x = cell_x + (cell_w - img.size[0]) // 2
                y = cell_y + (cell_h - img.size[1]) // 2
                paste_with_alpha(canvas, img, (x, y))
            return

        raise ValueError(f"Unsupported stack layout kind: {kind}")

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    def process(
        self,
        frames,
        region_position,
        region_size_px,
        face_distribution,
        interval_frames,
        overflow_mode,
        stack_direction,
        face_scale_pct,
        face_padding_px,
        face_gap_px,
        face_align_main,
        face_align_cross,
        chroma_r,
        chroma_g,
        chroma_b,
        face_sequence=None,
        face_images=None,
    ):
        """Main node execution."""
        if len(frames.shape) != 4:
            raise ValueError(
                "Input 'frames' must be a batch IMAGE tensor with shape [N, H, W, C]."
            )

        # Determine input mode
        use_sequence = face_sequence is not None
        use_image = face_images is not None

        if not use_sequence and not use_image:
            raise ValueError(
                "You must connect either 'face_sequence' (FACE_SEQUENCE) "
                "or 'face_images' (IMAGE) to the node."
            )

        # Pre-process legacy IMAGE input
        if use_image and not use_sequence:
            if len(face_images.shape) != 4 or face_images.shape[0] < 1:
                raise ValueError(
                    "Input 'face_images' must be a valid IMAGE batch with at least one image."
                )
            face_count = face_images.shape[0]
            faces_pil_legacy = [self._prepare_face(face_images[i]) for i in range(face_count)]
        else:
            faces_pil_legacy = []

        n, orig_h, orig_w, _ = frames.shape

        if region_position in ["left", "right"]:
            max_region_size = max(1, orig_w - 1)
        else:
            max_region_size = max(1, orig_h - 1)

        region_size_px = max(1, min(region_size_px, max_region_size))

        if region_position in ["left", "right"]:
            region_w = region_size_px
            region_h = orig_h
            video_max_w = orig_w - region_size_px
            video_max_h = orig_h
        else:
            region_w = orig_w
            region_h = region_size_px
            video_max_w = orig_w
            video_max_h = orig_h - region_size_px

        if video_max_w < 1 or video_max_h < 1:
            raise ValueError(
                "The reserved region size is too large for the current frame resolution."
            )

        fitted_video_w, fitted_video_h = fit_inside(orig_w, orig_h, video_max_w, video_max_h)

        chroma_rgba = (chroma_r, chroma_g, chroma_b, 255)

        out_frames = []

        for i in range(n):
            frame_pil_src = tensor_to_pil(frames[i]).convert("RGB")
            frame_pil = frame_pil_src.resize((fitted_video_w, fitted_video_h), Image.LANCZOS)

            canvas = Image.new("RGBA", (orig_w, orig_h), color=(0, 0, 0, 255))

            if region_position == "left":
                region_x, region_y = 0, 0
                video_x, video_y = region_size_px, (orig_h - fitted_video_h) // 2

            elif region_position == "right":
                region_x, region_y = orig_w - region_size_px, 0
                video_x, video_y = 0, (orig_h - fitted_video_h) // 2

            elif region_position == "top":
                region_x, region_y = 0, 0
                video_x, video_y = (orig_w - fitted_video_w) // 2, region_size_px

            else:  # bottom
                region_x, region_y = 0, orig_h - region_size_px
                video_x, video_y = (orig_w - fitted_video_w) // 2, 0

            region_img = Image.new("RGBA", (region_w, region_h), color=chroma_rgba)
            canvas.paste(region_img, (region_x, region_y))
            canvas.paste(frame_pil.convert("RGBA"), (video_x, video_y))

            # Resolve faces for this frame
            if use_sequence:
                faces_for_frame = self._resolve_faces_for_frame_sequence(
                    face_sequence=face_sequence,
                    frame_idx=i,
                    overflow_mode=overflow_mode,
                )
            else:
                faces_for_frame = self._resolve_faces_for_frame_image(
                    faces_pil=faces_pil_legacy,
                    frame_idx=i,
                    face_distribution=face_distribution,
                    interval_frames=interval_frames,
                    overflow_mode=overflow_mode,
                )

            if len(faces_for_frame) == 1:
                single_face = self._resize_single_face(
                    face=faces_for_frame[0],
                    region_w=region_w,
                    region_h=region_h,
                    face_scale_pct=face_scale_pct,
                    face_padding_px=face_padding_px,
                )

                self._paste_single_face(
                    canvas=canvas,
                    face=single_face,
                    region_x=region_x,
                    region_y=region_y,
                    region_w=region_w,
                    region_h=region_h,
                    region_position=region_position,
                    face_padding_px=face_padding_px,
                    face_align_main=face_align_main,
                    face_align_cross=face_align_cross,
                )

            else:
                effective_stack_direction = stack_direction
                if stack_direction == "auto":
                    effective_stack_direction = (
                        "vertical" if region_position in ["left", "right"] else "horizontal"
                    )

                self._paste_stack_faces(
                    canvas=canvas,
                    faces_pil=faces_for_frame,
                    region_x=region_x,
                    region_y=region_y,
                    region_w=region_w,
                    region_h=region_h,
                    face_scale_pct=face_scale_pct,
                    face_padding_px=face_padding_px,
                    face_gap_px=face_gap_px,
                    face_align_main=face_align_main,
                    face_align_cross=face_align_cross,
                    stack_direction=effective_stack_direction,
                )

            out_frames.append(pil_to_tensor(canvas.convert("RGB")))

        out = torch.stack(out_frames, dim=0)
        return (out,)


# ---------------------------------------------------------------------------
# ComfyUI registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "BFSAutoCropHeadReference": AutoCropHeadReference,
    "ReservedRegionFrameComposer": ReservedRegionFrameComposer,
    "FrameRangedFaceLoader": FrameRangedFaceLoader,
    "FaceSequenceBatch": FaceSequenceBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BFSAutoCropHeadReference": "BFS Auto Crop Head Reference",
    "ReservedRegionFrameComposer": "Reserved Region Frame Composer",
    "FrameRangedFaceLoader": "Frame Ranged Face Loader",
    "FaceSequenceBatch": "Face Sequence Batch",
}
