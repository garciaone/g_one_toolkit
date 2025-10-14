from inspect import cleandoc
import torch

class ImageReposition:
    """
    ImageReposition

    A node that repositions an image by shifting it along the X and Y axes.

    """
    CATEGORY = "g_one_toolkit/Image"

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required":{"images": ("IMAGE",),
                            "X_Shift": ("INT", {
                            "default": 0,
                            "min": -4096, 
                            "max": 4096, 
                            "step": 1, 
                            "display": "number" }),
                            "Y_Shift": ("INT", {
                            "default": 0,
                            "min": -4096, 
                            "max": 4096, 
                            "step": 1, 
                            "display": "number" }),
                            "custom_output": (["disable","enable"],),
                            "Out_Width": ("INT", {
                            "default": 512,
                            "min": 1, 
                            "max": 4096, 
                            "step": 1, 
                            "display": "number" }),
                            "Out_Height": ("INT", {
                            "default": 512,
                            "min": 1, 
                            "max": 4096, 
                            "step": 1, 
                            "display": "number" }),


                            }
        }
    
    
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "imageReposition"


    def imageReposition(self, images, X_Shift, Y_Shift, custom_output ,Out_Width, Out_Height):

        B, H, W, C = images.shape


        if custom_output == "disable":
            # canvas
            out = torch.zeros((B, H, W, C), dtype=images.dtype, device=images.device)  

        if custom_output == "enable":
            out = torch.zeros((B, Out_Height, Out_Width, C), dtype=images.dtype, device=images.device)  


        src_y0 = max(0, -Y_Shift)
        src_x0 = max(0, -X_Shift)
        dst_y0 = max(0,  Y_Shift)
        dst_x0 = max(0,  X_Shift)

        y_len  = max(0, min(H - src_y0, Out_Height - dst_y0))
        x_len  = max(0, min(W - src_x0, Out_Width - dst_x0))

        # paste
        if y_len > 0 and x_len > 0:
            out[:, dst_y0:dst_y0+y_len, dst_x0:dst_x0+x_len] = images[:, src_y0:src_y0+y_len, src_x0:src_x0+x_len]


        # B, H, W, C = images.shape
        # shifted = torch.zeros_like(images)

        # # Y direction
        # if Y_Shift >= 0:
        #     y_src = slice(Y_Shift, None)
        #     y_dst = slice(0, H - Y_Shift)
        # else:
        #     y_src = slice(0, H + Y_Shift)
        #     y_dst = slice(-Y_Shift, None)

        # # X direction
        # if X_Shift >= 0:
        #     x_src = slice(X_Shift, None)
        #     x_dst = slice(0, W - X_Shift)
        # else:
        #     x_src = slice(0, W + X_Shift)
        #     x_dst = slice(-X_Shift, None)

        # # assign shifted region
        # shifted[:, y_dst, x_dst] = images[:, y_src, x_src]

        return (out,)






# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "g-one Image Reposition" : ImageReposition
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "g-one Image Reposition" : "ImageReposition"
}