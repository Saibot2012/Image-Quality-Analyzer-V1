import cv2
import numpy as np

def create_overlay(img, heatmap):
    # 1. Resize heatmap to match image size
    heatmap_resized = cv2.resize(
        heatmap,
        (img.shape[1], img.shape[0])
    )

    # 2. Convert heatmap to 0–255 image
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)

    # 3. Apply color map (this makes it look like a real heatmap)
    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_INFERNO
    )

    # 4. Blend with original image
    blended = cv2.addWeighted(
        img, 0.6,
        heatmap_color, 0.4,
        0
    )

    return blended