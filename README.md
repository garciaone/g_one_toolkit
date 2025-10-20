# g_one_toolkit

A personal sandbox of custom Comfy UI nodes. 

**Note: Currently the nodes have been written as quick prototypes as opposed to production-ready code. Lots of testing, clean up, refactoring and error checking still needed**

# Nodes

## G-One Face Tracking Masks
**Description:** This node creates tracked masks of facial features such as the left eye, mouth, etc. The masks are mapped to a face mesh that is generated using the MediaPipe Face Landmarker model, allowing for either soft or hard-edge masks. Note: This is currently only configured to process one face.

| <img src=".\content\faceTrackingMasks\faceTrackingMasks_eyes.webp" width="100%"> | <img src=".\content\faceTrackingMasks\faceTrackingMasks_mouth.webp" width="100%"> | <img src=".\content\faceTrackingMasks\faceTrackingMasks_eyebrows.webp" width="90%"> |
|---|---|---|
| L_Eye & R_Eye / Simple-2D / feather_falloff = 30, grow_selection = 1 | Mouth / Simple-2D / feather_falloff = 30, grow_selection = 1 | L_Eyebrow & R_Eyebrow / Simple-2D / inner_falloff = 5 / feather_falloff = 20, grow_selection = 0|

### **Features**
 - Facial masks mapped to 3D mesh.
 - Multiple isolated facial masks.
 - Face tracking (MediaPipe Face Landmarker model)
 - Hard / Soft Masks
 - Mask region expansion 
 - Mask falloff options: 
    - "simple-2D": Fast method. Simple 2D falloff in pixel space. The core mask is still mapped in 3D space, but the falloff is in screen space. Best for front-on perspectives. 
    - "mesh-aware-3D": Slower method / **Experimental**. Creates a falloff that follows the contours of the face on the face mesh directly. Note: This has fairly sensitive range of small values that are stable, when extending beyond 0.05 flickering will occur. post_blur_2d can be used to help smooth out the flicker and is recommended when experiencing artefacts. 

### **Node Parameter**

| <img src=".\content\faceTrackingMasks\node_ui.png" width="70%"> | <img src=".\content\faceTrackingMasks\faceTrackingMasks_brow_ridge.webp" width="100%"> | <img src=".\content\faceTrackingMasks\faceTrackingMasks_brow_ridge_mask.webp" width="100%"> |
|---|---|---|
| Node UI | Brow Ridge / mesh-aware-3D / feather_falloff = .06, grow_selection = 1, post_blur_2d=7 | output_mode: mask|

- **Face Tracking (MediaPipe)**
    - **frame_rate**: Input FPS
    - **min_face_detection_confidence**: Minimum score for the face detector to accept a detection
    - **min_face_presence_confidence**: Minimum score for face presence from the landmark model.
    - **min_tracking_confidence**: Minimum score for tracking between frames to be considered successful.
- **Mask Selection**
    - **mask_selection**: "L_Eye", "R_Eye", "L_Eyebrow", "R_Eyebrow","Brow_Ridge","Nose", "Lips","Chin","Face"
    - **optional_selection**: "L_Eye", "R_Eye", "L_Eyebrow", "R_Eyebrow","Brow_Ridge","Nose", "Lips","Chin","Face"
    - **grow_selection**: Expands the selection to neighbouring points on the face mesh.
- **Falloff**
    - **falloff_mode**: "simple-2D", "mesh-aware-3D" – Note: "inner_falloff" and "feather_falloff" will need to be re-adjusted based on which falloff method is selected. "mesh-aware-3D" requires much smaller values, currently only stable with a feather_falloff <= 0.05, anything greater will require post_blur_2D to attempt to smooth out the flicker, generally post blur is recommended when experiencing artefacts while using mesh-aware-3D method. 
    - **inner_falloff**: The internal distance of the soft falloff; larger values create a softer mask. Note: Set to 0 for hard edge mask
    - **feather_falloff**: The external distance of the soft falloff; larger values create a softer mask.  Note: Set to 0 for hard edge mask
- **Post Operations**
    - **post_blur_2D**: An additional blur that can be used optionally to soften/smooth out the mask. Recommended to use in conjunction with the "mesh-aware-3D" falloff method.
    - **post_remap_in_min**: Minimal input value for a linear remap of the mask values. Default 0.
    - **post_remap_in_max**: Maximum input value for a linear remap of the mask values. Default 1.
    - **post_remap_out_min**: Minimal output value for a linear remap of the mask values. Default 0.
    - **post_remap_out_max**: Maximum output value for a linear remap of the mask values. Default 1.
- **Output & Visualisation**
    - **output_mode**: "mask", "visualisation" – Mask will generate black & white image(s). Visualisation will overlay the MediaPipe face mesh along with any mask being generated over the input image(s) for fine-tuning and debugging.
    - **vis_colour**: A selection of different colours used in visualisation mode.

### **To Do**
 - Testing
 - General clean-up of code and refactoring. 
 - Additional testing / unit testing needs to be set up. 
 - Explore more robust "mesh-aware-3D" alternatives to geometry based falloffs.
 - Update method of generating surfacing from face mesh verts. Currently, it uses a convex hull for simplicity; however, this causes issues when representing facial contours accurately.
 - All outputs are generated as images of type RGB BxWxHxC. An optional direct mask output type will be included.
 - Falloff settings need to be further normalised to ensure a less drastic shift in the output due to falloff method switching. 
 - Allow multiple face generation.

## G-One Face Tracking Features

**Description:** A simple wrapper for MediaPipe's Face Landmarker model. Draws MediaPipe FaceMesh features. Supports per-group toggles (left brow, right eyes lips, etc) with configurable thickness and optional vertex markers.Note: This is currently only configured to process one face.


| <img src=".\content\faceTrackingFeatures\faceTrackingFeatures_all.webp" width="100%"> | <img src=".\content\faceTrackingFeatures\faceTrackingFeatures_eyes_eyebrows.webp" width="100%"> | <img src=".\content\faceTrackingFeatures\faceTrackingFeatures_lips.webp" width="100%"> |
|---|---|---|
| output: visualisation / All Features | output: mask, selection: l_eye + l_eye_lid + l_eyebrow + r_eye + r_eye_lid + r_eyebrow | output: mask, selection: lips |

### **Features**
 - Facial features tracked to face mesh.
 - Selectable facial features.
 - Multiple eye render options.
 - Thickness control.

### **Node Parameter**
<img src=".\content\faceTrackingFeatures\node_ui.png" width="20%"> 


- **Face Tracking (MediaPipe)**
    - **frame_rate**: Input FPS
    - **min_face_detection_confidence**: Minimum score for the face detector to accept a detection
    - **min_face_presence_confidence**: Minimum score for face presence from the landmark model.
    - **min_tracking_confidence**: Minimum score for tracking between frames to be considered successful.
- **Feature Selection / Rendering**
    - The following parameters are available for: eye (L/R), eye lid (L/R), eyebrow (L/R), lips, face oval & face points.
    - **visibility toggle**: Turns feature on/off.
    - **thickness**: Controls how thick the lines are rendered. 
    - Specific eye controls:
        - **radius**: Controls the radius of a circle drawn around the centre of the pupil.
        - **fill**: Toggle to fill the circle or render the outline.
    - **face points radius**: Controls the size of the tracked face points. 
- **Output & Visualisation**
    - **output_mode**: "mask", "visualisation" – Mask will generate black & white image(s). Visualisation will overlay the face mesh along with any mask being generated over the input image(s) for fine-tuning and debugging.
    - **vis_colour**: A selection of different colours used in visualisation mode.

### **To Do**
 - General clean-up of code and refactoring. 
 - Additional testing / unit testing needs to be set up. 
 - Allow multiple face generation.
