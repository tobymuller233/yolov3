import cv2
import numpy as np

def resize_and_pad(image_path, output_path, target_size=640):
    # Load the image
    image = cv2.imread(image_path)
    original_height, original_width = image.shape[:2]

    # Calculate new dimensions while keeping aspect ratio
    scale = target_size / max(original_height, original_width)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Create a blank (zero-padded) canvas of the target size
    padded_image = np.zeros((target_size, target_size, 3), dtype=np.uint8)

    # Center the resized image on the canvas
    pad_x = (target_size - new_width) // 2
    pad_y = (target_size - new_height) // 2
    padded_image[pad_y:pad_y + new_height, pad_x:pad_x + new_width] = resized_image

    # Save the output
    cv2.imwrite(output_path, padded_image)

# Example usage
resize_and_pad("data/images/bus.jpg", "bus.jpg")