# =============================================================================
# Name:           track_face_masks.py
# Description:    MediaPipe FaceMesh utilities + ComfyUI node to generate hard/soft
#                 region masks (eyes, brows, lips, nose, chin, face oval) with
#                 optional mesh-aware falloff and visualization overlays.
# Author:         Beau Garcia
# Email:          garciaone@gmail.com
#
#  Functionality:
#  - Facial masks mapped to 3D mesh.
#  - Multiple isolated facial masks.
#  - Face tracking (MediaPipe Face Landmarker model)
#  - Hard / Soft Masks
#  - Mask region expansion 
#  - Mask falloff options: 
#     - "simple-2D": Fast method. Simple 2D falloff in pixel space. The core mask is still mapped in 3D space, but the falloff is in screen space. Best for front-on perspectives. 
#     - "mesh-aware-3D": Slower method. Creates a falloff that follows the contours of the face on the face mesh directly. Note: This method currently has some known bugs and is currently prone to artifacts.
# 
# To Do:
#  - General clean-up of code and refactoring. 
#  - Additional testing / unit testing needs to be set up. 
#  - Fix first frame pop, feather falloff and artifacts in mesh-aware-3D mode.
#  - Update method of generating surfacing from face mesh verts. Currently, it uses a convex hull for simplicity; however, this causes issues when representing facial contours accurately.
#  - All outputs are generated as images of type RGB BxWxHxC. An optional direct mask output type will be included.
#  - Mesh-aware-3D has limitations due to the face mesh topology produced by the MediaPipe Face Landmarker model, resulting in occasional artefacts in the mask. This can mostly be resolved by falloff settings and post-2D blur. More work can be done to make this option more robust overall. 
#  - Falloff settings need to be further normalised to ensure a less drastic shift in the output due to falloff method switching. 
#  - Allow multiple face generation.
#
# Dependencies:
#   - Python >= 3.10 (recommended)
#   - numpy
#   - opencv-python
#   - mediapipe (Tasks API)
#   - scipy (for scipy.spatial.Delaunay)
#   - torch (for ComfyUI IMAGE tensors)
#   - heapq (standard library)
# =============================================================================

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import torch
import heapq
from scipy.spatial import Delaunay
from pathlib import Path

mp_drawing = mp.solutions.drawing_utils
FM         = mp.solutions.face_mesh
DEBUG = False
############## Mediapipe Model Path

current_module_path = str(Path(__file__).resolve().parent)
model_path = current_module_path + "/mediaPipe_model/face_landmarker.task"

######################################################### Node Function
def trackFaceMasks(images,**kargs):

    frame_rate = kargs["frame_rate"]
    min_face_det_conf = kargs["min_face_detection_confidence"]
    min_face_pres_conf = kargs["min_face_presence_confidence"]
    min_track_conf = kargs["min_tracking_confidence"]
    falloff_mode = kargs["falloff_mode"]
    mask_selection = kargs["mask_selection"]
    optional_selection = kargs["optional_selection"]
    grow_selection = kargs["grow_selection"]
    inner_falloff = kargs["inner_falloff"]
    feather_falloff = kargs["feather_falloff"]
    post_blur = kargs["post_blur_2D"]
    output_mode = kargs["output_mode"]
    vis_colour = kargs["vis_colour"]
    post_remap_in_min = kargs["post_remap_in_min"]
    post_remap_in_max = kargs["post_remap_in_max"]
    post_remap_out_min = kargs["post_remap_out_min"]
    post_remap_out_max = kargs["post_remap_out_max"]
    TRIS = None

    ###### Detector setup 
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

    FM = mp.solutions.face_mesh
    IDX_L_EYE   = indices_from_connections(FM.FACEMESH_LEFT_EYE)
    IDX_R_EYE   = indices_from_connections(FM.FACEMESH_RIGHT_EYE)
    IDX_L_BROW  = indices_from_connections(FM.FACEMESH_LEFT_EYEBROW)
    IDX_R_BROW  = indices_from_connections(FM.FACEMESH_RIGHT_EYEBROW)
    IDX_LIPS    = indices_from_connections(FM.FACEMESH_LIPS)
    IDX_OVAL    = indices_from_connections(FM.FACEMESH_FACE_OVAL)
    IDX_NOSE    =  [ 6, 197, 195, 5, 4,1,19,94,419,248,281,275,274,354,370,456,363,440,457,461,462,196,3,51,45,44,125,141,236,134,220,237,241,141,360,456,420,429,279,168,278,455,305,290,328,198,131,49,99,240,235,48,236,456,122,351,174,399]
    IDX_CHIN    =  [199,175,396,171,208,428]
    IDX_BROW_RIDGE  =  [107,55,8,285,336,9,107]

    colour_select = {"White":[255,255,255],
                     "Red":[255,0,0],
                     "Green":[0,255,0],
                     "Blue":[0,0,255],
                     "Yellow":[255,255,0],
                     "Purple":[255,0,255],
                     "Cyan": [0,255,255]
                    }
    
    colourBGR =    [colour_select[vis_colour][2],
                colour_select[vis_colour][1],
                colour_select[vis_colour][0],
                ]
    
    mask_selections = {"None": None,
                       "L_Eye": IDX_L_EYE,
                       "R_Eye": IDX_R_EYE,
                       "L_Eyebrow": IDX_L_BROW,
                       "R_Eyebrow": IDX_R_BROW,
                       "Nose": IDX_NOSE,
                       "Lips": IDX_LIPS,
                       "Chin": IDX_CHIN, 
                       "Face": IDX_OVAL, 
                       "Brow_Ridge": IDX_BROW_RIDGE,
                       "Forehead": None,
                       "L_Cheek": None,
                       "R_Cheek": None
                      }

    mask_selection_set_0 = mask_selections[mask_selection]
    optional_selection_set_0 = mask_selections[optional_selection]

    #### Added to normalize values when switching fall off methods, but needs further experimentation. 
    if falloff_mode == "mesh-aware-3D":
        feather_falloff_n = feather_falloff
        inner_falloff_n = inner_falloff

    # Process frames in `images` (each HxWxC RGB tensor/array) 
    is_single = hasattr(images, "ndim") and images.ndim == 3
    frames = [images] if is_single else list(images)

    for frame_idx, frame in enumerate(frames):
        print(f"Tracking Face Masks : Processing frame {frame_idx+1}/{len(frames)}...")

        ### Reset the selection set on each frame , avoids compounding growth selections that animate over time. 
        mask_selection_set = mask_selection_set_0
        optional_selection_set = optional_selection_set_0

        if isinstance(frame, torch.Tensor):frame = frame.detach().cpu().numpy()

        if np.issubdtype(frame.dtype, np.floating):
            if frame.max() <= 1.0: frame = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
            else: frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)
        else: frame = frame.astype(np.uint8, copy=False)
        frame = np.ascontiguousarray(frame)  # HxWx3

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        ts_ms = int((frame_idx / frame_rate) * 1000)
        result = detector.detect_for_video(mp_image, ts_ms)

        if output_mode == "visualisation": bg_frame = frame
        else: bg_frame = np.full_like(frame, fill_value=(0, 0, 0)) ## Black background
        frame_bgr = cv2.cvtColor(bg_frame, cv2.COLOR_RGB2BGR) ## to BGR for OpenCV drawing
        frame_bgr_f32   = frame_bgr.astype(np.float32) # to float32 avoids uint8 wrap/rounding

        if result.face_landmarks:
            for face in result.face_landmarks:

                ##### simple-2D mode: use hardcoded 2D masks from landmark indices #####
                if falloff_mode == "simple-2D":

                    if mask_selection_set != None: ############## Main Mask Selection

                        if mask_selection == "Brow_Ridge": ### compound mask (UPDATE to non-concave hull method)
                            mask_selection_set_01 = mask_selection_set
                            mask_selection_set_02 = mask_selections["L_Eyebrow"]
                            mask_selection_set_03 = mask_selections["R_Eyebrow"]
                            if grow_selection != 0: ############## Grow Selection
                                mask_selection_set_01 = expand_selection(mask_selection_set_01, steps=grow_selection, n_vertices=len(face))
                                mask_selection_set_02 = expand_selection(mask_selection_set_02, steps=grow_selection, n_vertices=len(face))
                                mask_selection_set_03 = expand_selection(mask_selection_set_03, steps=grow_selection, n_vertices=len(face))

                            mask_S01 = hard_mask_from_indices_convex(frame_bgr.shape, face, mask_selection_set_01)
                            mask_S02 = hard_mask_from_indices_convex(frame_bgr.shape, face, mask_selection_set_02)
                            mask_S03 = hard_mask_from_indices_convex(frame_bgr.shape, face, mask_selection_set_03)
                            mask = np.maximum(mask_S01, mask_S02)
                            mask = np.maximum(mask, mask_S03)

                        else:
                            if grow_selection != 0: ############## Grow Selection
                                mask_selection_set = expand_selection(mask_selection_set, steps=grow_selection, n_vertices=len(face))

                            mask = hard_mask_from_indices_convex(frame_bgr.shape, face, mask_selection_set)

                        if inner_falloff + feather_falloff > 0.0: # soften edges when values are non-zero
                            mask = soften_mask(mask, inner_px=inner_falloff, feather_px=feather_falloff)
                        if post_blur > 0.0: mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=post_blur) 
                        mask_f32_0 = mask_convert_format(mask)

                    if optional_selection_set != None: ######### Opt Mask Selection

                        if optional_selection == "Brow_Ridge": ### compound mask (UPDATE to non-concave hull method)
                            optional_selection_set_01 = optional_selection_set
                            optional_selection_set_02 = mask_selections["L_Eyebrow"]
                            optional_selection_set_03 = mask_selections["R_Eyebrow"]

                            if grow_selection != 0: ############## Grow Selection
                                optional_selection_set_01 = expand_selection(optional_selection_set_01, steps=grow_selection, n_vertices=len(face))
                                optional_selection_set_02 = expand_selection(optional_selection_set_02, steps=grow_selection, n_vertices=len(face))
                                optional_selection_set_03 = expand_selection(optional_selection_set_03, steps=grow_selection, n_vertices=len(face))

                            mask_S01 = hard_mask_from_indices_convex(frame_bgr.shape, face, optional_selection_set_01)
                            mask_S02 = hard_mask_from_indices_convex(frame_bgr.shape, face, optional_selection_set_02)
                            mask_S03 = hard_mask_from_indices_convex(frame_bgr.shape, face, optional_selection_set_03)
                            mask = np.maximum(mask_S01, mask_S02)
                            mask = np.maximum(mask, mask_S03)

                        else:
                            if grow_selection != 0: ############## Grow Selection
                                optional_selection_set = expand_selection(optional_selection_set, steps=grow_selection, n_vertices=len(face))

                            mask = hard_mask_from_indices_convex(frame_bgr.shape, face, optional_selection_set)

                        if inner_falloff + feather_falloff > 0.0: # soften edges when values are non-zero
                            mask = soften_mask(mask, inner_px=inner_falloff, feather_px=feather_falloff)
                        if post_blur > 0.0: mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=post_blur) 
                        mask_f32_1 = mask_convert_format(mask)

                    if optional_selection_set != None and mask_selection_set != None:
                        mask_f32 = np.maximum(mask_f32_0, mask_f32_1)
                    else:
                        if mask_selection_set != None: mask_f32 = mask_f32_0
                        elif optional_selection_set != None: mask_f32 = mask_f32_1

                    if optional_selection_set != None or mask_selection_set != None:

                        if output_mode == "visualisation":
                            tint  = np.array([colourBGR[0], colourBGR[1], colourBGR[2]], dtype=frame_bgr_f32.dtype) 
                        else:
                            tint  = np.array([255, 255, 255], dtype=frame_bgr_f32.dtype) 

                        ### remap
                        mask_f32 = remap_linear(mask_f32, post_remap_in_min, post_remap_in_max, out_min=post_remap_out_min, out_max=post_remap_out_max, clip=True)
                        
                        alpha = 1                                             # blend strength
                        comp_out = frame_bgr_f32 * (1 - alpha*mask_f32) + tint * (alpha*mask_f32)
                        comp_out = comp_out.astype(frame_bgr.dtype)

                        frame_bgr = comp_out

                ##### mesh-aware-3D mode: use soft falloff masks from 3D mesh distances #####
                if falloff_mode == "mesh-aware-3D":

                    if TRIS is None and face: # init TRIS once
                        TRIS = build_tris_once(face)

                    if mask_selection_set != None: ############## Main Mask Selection

                        if mask_selection == "Brow_Ridge": ### compound mask (UPDATE to non-concave hull method)
                            mask_selection_set_01 = mask_selection_set
                            mask_selection_set_02 = mask_selections["L_Eyebrow"]
                            mask_selection_set_03 = mask_selections["R_Eyebrow"]
                            if grow_selection != 0: ############## Grow Selection
                                mask_selection_set_01 = expand_selection(mask_selection_set_01, steps=grow_selection, n_vertices=len(face))
                                mask_selection_set_02 = expand_selection(mask_selection_set_02, steps=grow_selection, n_vertices=len(face))
                                mask_selection_set_03 = expand_selection(mask_selection_set_03, steps=grow_selection, n_vertices=len(face))

                            mask_S01 = mesh_soft_falloff_mask(frame_bgr.shape, face, mask_selection_set_01, TRIS, inner=inner_falloff_n, feather=feather_falloff_n, space="3d")
                            mask_S02 = mesh_soft_falloff_mask(frame_bgr.shape, face, mask_selection_set_02, TRIS, inner=inner_falloff_n, feather=feather_falloff_n, space="3d")
                            mask_S03 = mesh_soft_falloff_mask(frame_bgr.shape, face, mask_selection_set_03, TRIS, inner=inner_falloff_n, feather=feather_falloff_n, space="3d")

                            mask = np.maximum(mask_S01, mask_S02)
                            mask = np.maximum(mask, mask_S03)

                        else:
                            if grow_selection != 0: ############## Grow Selection
                                mask_selection_set = expand_selection(mask_selection_set, steps=grow_selection)

                            mask = mesh_soft_falloff_mask(frame_bgr.shape, face, mask_selection_set, TRIS, inner=inner_falloff_n, feather=feather_falloff_n, space="3d")
                            
                        ########################## Eye Fix
                        if mask_selection == "L_Eye" or mask_selection == "R_Eye":
                            mask_eye_fix = hard_mask_from_indices_convex(frame_bgr.shape, face, mask_selection_set)
                            mask = np.maximum(mask_eye_fix, mask)

                        if post_blur > 0.0: mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=post_blur) 
                        mask_f32_0 = mask_convert_format(mask)

                    if optional_selection_set != None: ############## Opt Mask Selection

                        if optional_selection == "Brow_Ridge": ### compound mask (UPDATE to non-concave hull method)
                            optional_selection_set_01 = optional_selection_set
                            optional_selection_set_02 = mask_selections["L_Eyebrow"]
                            optional_selection_set_03 = mask_selections["R_Eyebrow"]
                            if grow_selection != 0: ############## Grow Selection
                                optional_selection_set_01 = expand_selection(optional_selection_set_01, steps=grow_selection, n_vertices=len(face))
                                optional_selection_set_02 = expand_selection(optional_selection_set_02, steps=grow_selection, n_vertices=len(face))
                                optional_selection_set_03 = expand_selection(optional_selection_set_03, steps=grow_selection, n_vertices=len(face))

                            mask_S01 = mesh_soft_falloff_mask(frame_bgr.shape, face, optional_selection_set_01, TRIS, inner=inner_falloff_n, feather=feather_falloff_n, space="3d")
                            mask_S02 = mesh_soft_falloff_mask(frame_bgr.shape, face, optional_selection_set_02, TRIS, inner=inner_falloff_n, feather=feather_falloff_n, space="3d")
                            mask_S03 = mesh_soft_falloff_mask(frame_bgr.shape, face, optional_selection_set_03, TRIS, inner=inner_falloff_n, feather=feather_falloff_n, space="3d")

                            mask = np.maximum(mask_S01, mask_S02)
                            mask = np.maximum(mask, mask_S03)

                        else:
                            if grow_selection != 0: ############## Grow Selection
                                optional_selection_set = expand_selection(optional_selection_set, steps=grow_selection)

                            mask = mesh_soft_falloff_mask(frame_bgr.shape, face, optional_selection_set, TRIS, inner=inner_falloff_n, feather=feather_falloff_n, space="3d")
                            
                        ########################## Eye Fix
                        if optional_selection == "L_Eye" or optional_selection == "R_Eye":
                            mask_eye_fix = hard_mask_from_indices_convex(frame_bgr.shape, face, optional_selection_set)
                            mask = np.maximum(mask_eye_fix, mask)
                        
                        if post_blur > 0.0: mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=post_blur) 
                        mask_f32_1 = mask_convert_format(mask)

                    if optional_selection_set != None and mask_selection_set != None:
                        mask_f32 = np.maximum(mask_f32_0, mask_f32_1)
                    else:
                        if mask_selection_set != None: mask_f32 = mask_f32_0
                        elif optional_selection_set != None: mask_f32 = mask_f32_1

                    if optional_selection_set != None or mask_selection_set != None:

                        if output_mode == "visualisation":
                            tint  = np.array([colourBGR[0], colourBGR[1], colourBGR[2]], dtype=frame_bgr_f32.dtype) 
                        else:
                            tint  = np.array([255, 255, 255], dtype=frame_bgr_f32.dtype) 

                        ### remap
                        mask_f32 = remap_linear(mask_f32, post_remap_in_min, post_remap_in_max, out_min=post_remap_out_min, out_max=post_remap_out_max, clip=True)
                       
                        alpha = 1                                             # blend strength
                        comp_out = frame_bgr_f32 * (1 - alpha*mask_f32) + tint * (alpha*mask_f32)
                        comp_out = comp_out.astype(frame_bgr.dtype)

                        frame_bgr = comp_out

                ### draw mesh if in visualization mode
                if output_mode == "visualisation":
                    face_proto = landmark_pb2.NormalizedLandmarkList()

                    face_proto.landmark.extend(
                        [landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in face]
                    )
                    solutions.drawing_utils.draw_landmarks(
                        image=frame_bgr,
                        landmark_list=face_proto,
                        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(

                            color=([colourBGR[0], colourBGR[1], colourBGR[2]]), # BGR colour
                            thickness=1,                           # line thickness
                            circle_radius=0                        # unused for lines
                        )
                    )

                    if DEBUG: draw_landmark_indices(frame_bgr, face, subset=None, stride=1,color=(255,0,255),dot_radius=1,font_scale=.2, thickness=1)

        # to sRGB for output
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

######################################################### UTL Functions
def indices_from_connections(conns):
    idx = set()
    for a, b in conns:
        idx.add(a); idx.add(b)
    return sorted(idx)

def remap_linear(x, in_min, in_max, out_min=0.0, out_max=1.0, clip=True):
    """
    Linearly maps values from [in_min, in_max] -> [out_min, out_max].
    Works with HxW or HxWxC arrays (float or uint8). Returns float32.
    """
    x = x.astype(np.float32, copy=False)
    denom = max(1e-12, float(in_max) - float(in_min))
    y = (x - float(in_min)) / denom
    if clip:
        y = np.clip(y, 0.0, 1.0)
    y = y * (float(out_max) - float(out_min)) + float(out_min)
    return y

def hard_mask_from_ordered_indices(frame_shape, landmarks, ordered_idx):
    h, w = frame_shape[:2]
    pts = np.array([(int(landmarks[i].x*w), int(landmarks[i].y*h)) for i in ordered_idx], np.int32)
    if len(pts) < 3: 
        mask = np.zeros((h,w), np.uint8); 
        if len(pts): cv2.circle(mask, tuple(pts[0]), 3, 255, -1, cv2.LINE_AA)
        return mask.astype(np.float32)/255.0
    pts = pts[:-1] if np.all(pts[0] == pts[-1]) else pts  # drop duplicate close
    mask = np.zeros((h,w), np.uint8)
    cv2.fillPoly(mask, [pts], 255)  # handles concave loops
    return mask.astype(np.float32)/255.0

def hard_mask_from_indices_convex(frame_shape, landmarks, indices):

    h, w = frame_shape[:2]
    pts = np.array([(int(landmarks[i].x*w), int(landmarks[i].y*h)) for i in indices], np.int32)
    if len(pts) < 3:  # fallback small dot
        print("hard_mask_from_indices: less than 3 points, using dot fallback")
        mask = np.zeros((h,w), np.uint8)
        if len(pts): cv2.circle(mask, tuple(pts[0]), 3, 255, -1, cv2.LINE_AA)
        return mask.astype(np.float32)/255.0
    hull = cv2.convexHull(pts)  # or use pts in correct order if you have it
    mask = np.zeros((h,w), np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask.astype(np.float32)/255.0

def build_tris_once(landmarks):
    # landmarks: one face (result.face_landmarks[0])
    P = np.array([[lm.x, lm.y] for lm in landmarks], np.float32)
    tri = Delaunay(P)
    return tri.simplices.copy()

def soften_mask(mask01, inner_px=0, feather_px=40):
    # mask01 is float32 in [0,1]
    binmask = (mask01*255).astype(np.uint8)
    dist_in  = cv2.distanceTransform(binmask, cv2.DIST_L2, 3)
    dist_out = cv2.distanceTransform(255 - binmask, cv2.DIST_L2, 3)
    # signed distance (negative inside)
    sdf = dist_out - dist_in
    t = (sdf + inner_px) / max(1e-6, feather_px)
    t = np.clip(t, 0, 1)
    t = t*t*(3 - 2*t) # smoothstep
    return 1.0 - t

def chin_indices(landmarks, idx_oval, mouth_center_idx=13): 
    ys = np.array([landmarks[i].y for i in idx_oval])
    y_thresh = landmarks[mouth_center_idx].y
    return [i for i in idx_oval if landmarks[i].y > y_thresh]

def forehead_indices(landmarks, idx_oval, brow_idx):
    brow_y = np.median([landmarks[i].y for i in brow_idx])
    return [i for i in idx_oval if landmarks[i].y < brow_y]


NOSE_SEEDS = [168, 6, 197, 195, 5, 4]
RIGHT_IRIS = [468,469,470,471,472]
LEFT_IRIS  = [473,474,475,476,477]

def mesh_soft_falloff_mask(frame_shape, landmarks, seed_indices,
                                 TRIS, inner=0.03, feather=0.06, space="3d"):
    H, W = frame_shape[:2]
    # coords for distance metric
    if space == "3d":
        P = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], np.float32)  
    else:
        P = np.array([[lm.x*W, lm.y*H] for lm in landmarks], np.float32)    

    # pixel coords for rasterization (keep float)
    Ppx = np.array([[lm.x*W, lm.y*H] for lm in landmarks], np.float32)      

    # adjacency from tessellation (build once per run)
    N = len(landmarks)
    adj = [[] for _ in range(max(N, 478))]
    for a,b in FM.FACEMESH_TESSELATION:
        if a < N and b < N:
            adj[a].append(b); adj[b].append(a)

    # Dijkstra with edge length weights
    dist = np.full(N, np.inf, np.float32)
    pq = []
    for s in seed_indices:
        if 0 <= s < N:
            dist[s] = 0.0
            heapq.heappush(pq, (0.0, s))
    while pq:
        d,u = heapq.heappop(pq)
        if d != dist[u]: continue
        Pu = P[u]
        for v in adj[u]:
            if v >= N: continue
            nd = d + float(np.linalg.norm(P[v] - Pu))
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))

    # smoothstep falloff
    t  = (dist - float(inner)) / max(1e-6, float(feather))
    t  = np.clip(t, 0.0, 1.0)
    wv = (1.0 - (t*t*(3.0 - 2.0*t))).astype(np.float32)  # per-vertex weights

    mask = np.zeros((H, W), np.float32)
    eps  = 1e-6
    for i,j,k in TRIS:
        x1,y1 = Ppx[i]; x2,y2 = Ppx[j]; x3,y3 = Ppx[k]
        xmin = int(max(min(x1,x2,x3), 0)); xmax = int(min(max(x1,x2,x3), W-1))
        ymin = int(max(min(y1,y2,y3), 0)); ymax = int(min(max(y1,y2,y3), H-1))
        denom = ( (y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3) )
        if abs(denom) < 1e-9: continue
        v1, v2, v3 = wv[i], wv[j], wv[k]
        for y in range(ymin, ymax+1):
            for x in range(xmin, xmax+1):
                w1 = ( (y2 - y3)*(x - x3) + (x3 - x2)*(y - y3) ) / denom
                w2 = ( (y3 - y1)*(x - x3) + (x1 - x3)*(y - y3) ) / denom
                w3 = 1.0 - w1 - w2
                if w1 >= -eps and w2 >= -eps and w3 >= -eps:
                    v = w1*v1 + w2*v2 + w3*v3
                    if v > mask[y, x]:
                        mask[y, x] = v

    return np.clip(mask, 0.0, 1.0)


def mask_convert_format(frame):
    frame_f32 = frame.astype(np.float32) # to float32 / Expected range is [0..1] per pixel.
    frame_f32_02  = frame_f32[..., None] #  H×W to H×W×1. This lets NumPy broadcast the mask across the 3 color channels when you multiply/add with an image (H×W×3).
    return frame_f32_02

def draw_landmark_indices(frame_bgr, face_landmarks,
                          subset=None,      # set/list of indices to show; None = all
                          stride=1,        # show every Nth index to reduce clutter
                          color=(0,255,255),
                          dot_radius=1,
                          font_scale=None, # auto if None
                          thickness=1):
    h, w = frame_bgr.shape[:2]
    if font_scale is None:
        font_scale = max(0.3, min(w, h) / 900.0)  # auto-scale text

    for i, lm in enumerate(face_landmarks):
        if subset and i not in subset: 
            continue
        if (i % stride) != 0:
            continue

        x, y = int(lm.x * w), int(lm.y * h)
        if x < 0 or y < 0 or x >= w or y >= h: 
            continue

        # dot
        cv2.circle(frame_bgr, (x, y), dot_radius, color, -1, cv2.LINE_AA)
        # label with small outline for readability
        label = str(i)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.putText(frame_bgr, label, (x+2, y-2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness+2, cv2.LINE_AA)
        cv2.putText(frame_bgr, label, (x+2, y-2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

    return frame_bgr

# Build (and cache) adjacency from the tessellation once
_ADJ = None
def _adjacency(n=478):
    global _ADJ
    if _ADJ is None:
        FM = mp.solutions.face_mesh
        adj = [[] for _ in range(n)]
        for a, b in FM.FACEMESH_TESSELATION:
            if a < n and b < n:
                adj[a].append(b); adj[b].append(a)
        _ADJ = adj
    return _ADJ

def expand_selection(seeds, steps=1, n_vertices=478):
    """
    Return indices within `steps` edge-hops of `seeds` on the FaceMesh graph.
    steps=0 -> just seeds. steps=1 -> direct neighbors, etc.
    """
    adj = _adjacency(n_vertices)
    visited = set(i for i in seeds if 0 <= i < n_vertices)
    frontier = set(visited)
    for _ in range(int(max(0, steps))):
        nxt = set()
        for u in frontier:
            for v in adj[u]:
                if v < n_vertices and v not in visited:
                    nxt.add(v)
        visited |= nxt
        frontier = nxt
        if not frontier:
            break
    return sorted(visited)