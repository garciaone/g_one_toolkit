# =============================================================================
# Name:           track_face_features.py
# Description:    Draws MediaPipe FaceMesh features on image frames (ComfyUI node).
#                 Supports per-group toggles (brows, lids, lips, face oval, irises)
#                 with configurable colors/thickness, and optional vertex markers.
# Author:         Beau Garcia
# Email:          garciaone@gmail.com
#
# Functionality:
#   - MediaPipe Tasks (FaceLandmarker) in VIDEO mode with per-frame timestamps
#   - Configurable render groups via RENDER_CFG:
#       * Left/Right eyebrows, Left/Right eyelids, Lips, Face oval
#       * Irises as fitted circles (radius scale / fill / thickness)
#       * Optional vertex dots (tracking markers)
#   - Converts ComfyUI IMAGE tensors/arrays to uint8 RGB for detection
#   - Overlays on original image or black background (background_image_vis)
#   - Returns processed frames in the original IMAGE tensor structure
#
# To Do:
#   - [ ] Add temporal smoothing (EMA) for landmarks to reduce jitter
#   - [ ] Cache detector instance across calls (avoid re-creating per run)
#   - [ ] Multi-face support (iterate & draw up to N faces)
#   - [ ] Expose color pickers / presets via UI or hex parsing
#   - [ ] Optional tessellation/contours overlay toggle
#
# Dependencies:
#   - Python >= 3.10 (recommended)
#   - numpy
#   - opencv-python
#   - mediapipe (Tasks API)
#   - torch (for ComfyUI IMAGE tensors)
# =============================================================================


import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import torch
from pathlib import Path

############## Mediapipe Model Path

current_module_path = str(Path(__file__).resolve().parent  )
model_path = current_module_path + "/mediaPipe_model/face_landmarker.task"


# ---------- MediaPipe aliases ----------
mp_drawing = mp.solutions.drawing_utils
FM         = mp.solutions.face_mesh


# Iris landmark indices
RIGHT_IRIS = [468, 469, 470, 471, 472]
LEFT_IRIS  = [473, 474, 475, 476, 477]


# ---------- Per-group render settings ----------
RENDER_CFG = {
    # Face shape (jaw/oval)
    "face_oval": {"on": False,  "thickness": 2, "color": (255, 100, 100)},
    # Eyebrows
    "l_brow":    {"on": True,  "thickness": 2, "color": (  0, 255,   255)},
    "r_brow":    {"on": True,  "thickness": 2, "color": (  0, 255,   255)},
    # Eyelids (eye contours)
    "l_eye":     {"on": True,  "thickness": 2, "color": (255, 0,   0)},
    "r_eye":     {"on": True,  "thickness": 2, "color": (255, 0,   0)},
    # Lips
    "lips":      {"on": True,  "thickness": 2, "color": (  50, 255, 150)},
    # Irises (draw as circles)
    "l_iris":    {"on": True,  "thickness": 2, "color": (  255, 100, 0),
                  "radius_scale": .025, "min_radius": 2, "fill": False},
    "r_iris":    {"on": True,  "thickness": 2, "color": (  255, 100, 0),
                  "radius_scale": .025, "min_radius": 2, "fill": False},
    # Tracking markers (all vertices)
    "verts":     {"on": False, "radius": 1,    "color": ( 255, 255,  50)},
}

def _fit_circle_px(face_landmarks, indices, w, h):
    pts = [(int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)) for i in indices]
    (cx, cy), r = cv2.minEnclosingCircle(np.array(pts, dtype=np.int32))
    return (int(cx), int(cy)), float(r)

def draw_face_features(frame_bgr, face_landmarks, cfg=RENDER_CFG):
    """
    face_landmarks: iterable of NormalizedLandmark
    cfg: dict like RENDER_CFG above
    """
    h, w = frame_bgr.shape[:2]

    # Optional vertex dots (tracking markers)
    if cfg["verts"]["on"]:
        r = int(cfg["verts"]["radius"])
        col = cfg["verts"]["color"]
        for lm in face_landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame_bgr, (x, y), max(r, 1), col, -1, cv2.LINE_AA)

    # Build protobuf once for drawing_utils
    face_proto = landmark_pb2.NormalizedLandmarkList()
    face_proto.landmark.extend(
        [landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in face_landmarks]
    )

    # Eyebrows
    if cfg["l_brow"]["on"]:
        mp_drawing.draw_landmarks(
            frame_bgr, face_proto, FM.FACEMESH_LEFT_EYEBROW,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=cfg["l_brow"]["color"], thickness=int(cfg["l_brow"]["thickness"])
            ),
        )
    if cfg["r_brow"]["on"]:
        mp_drawing.draw_landmarks(
            frame_bgr, face_proto, FM.FACEMESH_RIGHT_EYEBROW,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=cfg["r_brow"]["color"], thickness=int(cfg["r_brow"]["thickness"])
            ),
        )

    # Eyelids
    if cfg["l_eye"]["on"]:
        mp_drawing.draw_landmarks(
            frame_bgr, face_proto, FM.FACEMESH_LEFT_EYE,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=cfg["l_eye"]["color"], thickness=int(cfg["l_eye"]["thickness"])
            ),
        )
    if cfg["r_eye"]["on"]:
        mp_drawing.draw_landmarks(
            frame_bgr, face_proto, FM.FACEMESH_RIGHT_EYE,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=cfg["r_eye"]["color"], thickness=int(cfg["r_eye"]["thickness"])
            ),
        )

    # Lips
    if cfg["lips"]["on"]:
        mp_drawing.draw_landmarks(
            frame_bgr, face_proto, FM.FACEMESH_LIPS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=cfg["lips"]["color"], thickness=int(cfg["lips"]["thickness"])
            ),
        )

    # Face shape (oval)
    if cfg["face_oval"]["on"]:
        mp_drawing.draw_landmarks(
            frame_bgr, face_proto, FM.FACEMESH_FACE_OVAL,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=cfg["face_oval"]["color"], thickness=int(cfg["face_oval"]["thickness"])
            ),
        )

    # Irises as circles (L/R independently)
    if cfg["l_iris"]["on"]:
        c, r = _fit_circle_px(face_landmarks, LEFT_IRIS, w, h)
        r *= float(cfg["l_iris"]["radius_scale"])
        r = max(r, float(cfg["l_iris"]["min_radius"]))
        thick = -1 if cfg["l_iris"]["fill"] else int(cfg["l_iris"]["thickness"])
        cv2.circle(frame_bgr, c, int(r), cfg["l_iris"]["color"], thick, cv2.LINE_AA)

    if cfg["r_iris"]["on"]:
        c, r = _fit_circle_px(face_landmarks, RIGHT_IRIS, w, h)
        r *= float(cfg["r_iris"]["radius_scale"])
        r = max(r, float(cfg["r_iris"]["min_radius"]))
        thick = -1 if cfg["r_iris"]["fill"] else int(cfg["r_iris"]["thickness"])
        cv2.circle(frame_bgr, c, int(r), cfg["r_iris"]["color"], thick, cv2.LINE_AA)


def trackFaceFeatures(images,**kargs):

    frame_rate = kargs["frame_rate"]
    min_face_det_conf = kargs["min_face_detection_confidence"]
    min_face_pres_conf = kargs["min_face_presence_confidence"]
    min_track_conf = kargs["min_tracking_confidence"]
    background_image_vis = kargs["background_image_vis"]

    RENDER_CFG["l_eye"].update({"on": kargs["l_eye_lid_vis"], "thickness": kargs["eye_lid_thickness"]})
    RENDER_CFG["r_eye"].update({"on": kargs["r_eye_lid_vis"], "thickness": kargs["eye_lid_thickness"]})
    RENDER_CFG["l_iris"].update({"on": kargs["l_eye_vis"], "thickness": kargs["eye_thickness"]})
    RENDER_CFG["r_iris"].update({"on": kargs["r_eye_vis"], "thickness": kargs["eye_thickness"]})
    RENDER_CFG["r_iris"].update({"on": kargs["r_eye_vis"], "thickness": kargs["eye_thickness"]})
    RENDER_CFG["r_iris"].update({"fill": kargs["eye_fill"], "radius_scale": kargs["eye_radius"]})
    RENDER_CFG["l_iris"].update({"fill": kargs["eye_fill"], "radius_scale": kargs["eye_radius"]})
    
    RENDER_CFG["l_brow"].update({"on": kargs["l_eye_brow_vis"], "thickness": kargs["eye_brow_thickness"]})
    RENDER_CFG["r_brow"].update({"on": kargs["r_eye_brow_vis"], "thickness": kargs["eye_brow_thickness"]})

    RENDER_CFG["lips"].update({"on": kargs["lips_vis"], "thickness": kargs["lips_thickness"]})

    RENDER_CFG["face_oval"].update({"on": kargs["face_oval_vis"], "thickness": kargs["face_oval_thickness"]}) 

    RENDER_CFG["verts"].update({"on": kargs["face_points_vis"], "radius": kargs["face_points_radius"]}) 

    # ----------- Detector setup -----------
    base_options = python.BaseOptions(model_asset_path = model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        min_face_detection_confidence=min_face_det_conf,
        min_face_presence_confidence=min_face_pres_conf,
        min_tracking_confidence=min_track_conf,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        running_mode=vision.RunningMode.VIDEO,
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    # ----------- Process frames in `images` (each HxWxC RGB tensor/array) -----------
    is_single = hasattr(images, "ndim") and images.ndim == 3
    frames = [images] if is_single else list(images)

    for frame_idx, frame in enumerate(frames):
        print(f"Processing frame {frame_idx+1}/{len(frames)}...")

        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()

        if np.issubdtype(frame.dtype, np.floating):
            if frame.max() <= 1.0:
                frame = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8, copy=False)
        
        frame = np.ascontiguousarray(frame)  # HxWx3


        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        ts_ms = int((frame_idx / frame_rate) * 1000)
        result = detector.detect_for_video(mp_image, ts_ms)

        if background_image_vis:
            bg_frame = frame
        else:
            bg_frame = np.full_like(frame, fill_value=(0, 0, 0))

        frame_bgr = cv2.cvtColor(bg_frame, cv2.COLOR_RGB2BGR)

        if result.face_landmarks:
            for face in result.face_landmarks:
                draw_face_features(frame_bgr, face, cfg=RENDER_CFG)

        out_rgb_u8 = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames[frame_idx] = (out_rgb_u8.astype(np.float32) / 255.0)

    # write back in the same structure and return
    if is_single:
        if isinstance(images, torch.Tensor):
            images.copy_(torch.from_numpy(frames[0]).to(images.device).to(images.dtype))
        else:
            images[:] = frames[0]
    else:
        for i in range(len(images)):
            if isinstance(images[i], torch.Tensor):
                images[i].copy_(torch.from_numpy(frames[i]).to(images[i].device).to(images[i].dtype))
            else:
                images[i][:] = frames[i]

    return images