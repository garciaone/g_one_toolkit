# =============================================================================
# Name:           g_one_toolkit_nodes.py
# Description:    ComfyUI custom nodes:
#                   1) FaceTrackingMasks – generate region masks mapped to MediaPipe face mesh.
#                   2) FaceTrackingFeatures – draw MediaPipe FaceMesh features
# Author:         Beau Garcia
# Email:          garciaone@gmail.com
#
# Nodes:
#   - FaceTrackingMasks:
#   - FaceTrackingFeatures
#
# To Do:
#   - [ ] Error Checking
#   - [ ] General clean up

#
# Dependencies:
#   - Python >= 3.10 (recommended)
#   - numpy
#   - torch
#   - opencv-python
#   - mediapipe (Tasks API)
#   - ComfyUI (node hosting environment)
# =============================================================================

################ Imports ################
from .lib import face_tracking_features_node as ftf_node
from .lib import face_tracking_masks_node as ftm_node

################ Node 2 FaceTracking Masks ################
class FaceTrackingMasks:
    """
    FaceTracking
    
    """
    CATEGORY = "g_one_toolkit"

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required":{"images": ("IMAGE",),
                            ### Input FPS
                            "frame_rate": ("FLOAT", {"default": 25.00, "min": 1, "step": 0.01}),
                            ### MediaPipe Face Mesh parameters
                            "min_face_detection_confidence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                            "min_face_presence_confidence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                            "min_tracking_confidence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                            ### Mask Selection
                            "mask_selection": (["None", "L_Eye", "R_Eye", "L_Eyebrow", "R_Eyebrow","Brow_Ridge","Nose", "Lips","Chin","Face"], {"default": "None"}),
                            "optional_selection": (["None", "L_Eye", "R_Eye", "L_Eyebrow", "R_Eyebrow","Brow_Ridge","Nose", "Lips","Chin","Face"], {"default": "None"}),
                            "grow_selection": ("INT", {"default": 0, "min": 0, "step": 1}),
                            ### Fall off
                            "falloff_mode": (["simple-2D", "mesh-aware-3D"], {"default": "simple-2D"}),
                            "inner_falloff": ("FLOAT", {"default": .1, "min": -20,"max": 20, "step": 0.01}),
                            "feather_falloff": ("FLOAT", {"default": .1, "min": 0, "step": 0.01}),
                            ### Post Blur / Smooth
                            "post_blur_2D": ("FLOAT", {"default": .1, "min": 0, "step": 0.01}),
                            "post_remap_in_min": ("FLOAT", {"default": 0, "min": 0, "step": 0.01}),
                            "post_remap_in_max": ("FLOAT", {"default": 1, "min": 0, "step": 0.01}),
                            "post_remap_out_min": ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.01}),
                            "post_remap_out_max": ("FLOAT", {"default": 1, "min": 0, "max": 1,"step": 0.01}),
                            ### output options
                            "output_mode": (["mask", "visualisation"], {"default": ""}),
                            "vis_colour": (["White","Red","Green","Blue","Yellow","Purple","Cyan"], {"default": "Yellow"}),
                            }}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "trackFaceMasks"

    def trackFaceMasks(self, images, **kargs):
        images_tmp = images.clone()    
        images_tmp = ftm_node.trackFaceMasks(images_tmp, **kargs)
        return (images_tmp,)

################ Node: FaceTracking Features ################
class FaceTrackingFeatures:
    """
    FaceTracking
    A wrapper for MediaPipe Face Landmarks that detects and tracks facial landmarks in images or video frames.
    """
    CATEGORY = "g_one_toolkit"

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required":{"images": ("IMAGE",),
                            ### Input FPS
                            "frame_rate": ("FLOAT", {"default": 25.00, "min": 1, "step": 0.01}),
                            ### MediaPipe Face Mesh parameters
                            "min_face_detection_confidence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                            "min_face_presence_confidence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                            "min_tracking_confidence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                            # eye lids
                            "l_eye_lid_vis": ("BOOLEAN", {"default": True}),
                            "r_eye_lid_vis": ("BOOLEAN", {"default": True}),
                            "eye_lid_thickness": ("INT", {"default": 2, "min": 1, "max": 20, "step": 1}),
                            # eyes
                            "l_eye_vis": ("BOOLEAN", {"default": True}),
                            "r_eye_vis": ("BOOLEAN", {"default": True}),
                            "eye_thickness": ("INT", {"default": 2, "min": 1, "max": 20, "step": 1}),
                            "eye_radius": ("FLOAT", {"default": .25, "min": 0.0, "max": 20.0, "step": 0.01}),
                            "eye_fill": ("BOOLEAN", {"default": True}),
                            # eyebrows
                            "l_eyebrow_vis": ("BOOLEAN", {"default": True}),
                            "r_eyebrow_vis": ("BOOLEAN", {"default": True}),
                            "eyebrow_thickness": ("INT", {"default": 2, "min": 1, "max": 20, "step": 1}),
                            # lips
                            "lips_vis": ("BOOLEAN", {"default": True}),
                            "lips_thickness": ("INT", {"default": 2, "min": 1, "max": 20, "step": 1}),
                            # face
                            "face_oval_vis": ("BOOLEAN", {"default": True}),
                            "face_oval_thickness": ("INT", {"default": 2, "min": 1, "max": 20, "step": 1}),
                            # face points
                            "face_points_vis": ("BOOLEAN", {"default": True}),
                            "face_points_radius": ("INT", {"default": 2, "min": 1, "max": 20, "step": 1}),
                            ### output options
                            "output_mode": (["mask", "visualisation"], {"default": ""}),
                            }}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "trackFaceFeatures"

    def trackFaceFeatures(self, images, **kargs):
        images_tmp = images.clone()    
        images_tmp = ftf_node.trackFaceFeatures(images_tmp, **kargs)
        return (images_tmp,)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "g-one Face Tracking Features" : FaceTrackingFeatures,
    "g-one Face Tracking Masks" : FaceTrackingMasks,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "g-one Face Tracking Features" : "G-One Face Tracking Features",
    "g-one Face Tracking Masks" : "G-One Face Tracking Masks",
}