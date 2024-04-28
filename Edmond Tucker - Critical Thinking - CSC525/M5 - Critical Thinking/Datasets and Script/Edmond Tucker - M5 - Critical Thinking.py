import os
import cv2
import numpy as np
from PIL import Image, ImageOps


def augment_image(image_path, save_dir):
    img = Image.open(image_path).convert('RGB') 
    base_name = os.path.basename(image_path).split('_jpg')[0]
    img.save(os.path.join(save_dir, f'Original_{base_name}.jpg'))
    img_np = np.array(img)
    
    # Rotations
    rotations = [90, 180, 270]
    for angle in rotations:
        rotated_img = img.rotate(angle)
        rotated_img.save(os.path.join(save_dir, f'Rotated_{angle}_{base_name}.jpg'))
    
    # Horizontal / Vertical Flips
    h_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
    h_flip.save(os.path.join(save_dir, f'Flip.H_{base_name}.jpg'))
    v_flip = img.transpose(Image.FLIP_TOP_BOTTOM)
    v_flip.save(os.path.join(save_dir, f'Flip.V_{base_name}.jpg'))

    # Color
    inverted_image = ImageOps.invert(img)
    inverted_image.save(os.path.join(save_dir, f'Inverted_{base_name}.jpg'))

    # Brightness
    for i in range(1, 3):
        brightness_image = cv2.convertScaleAbs(img_np, alpha=1, beta=i*50)
        cv2.imwrite(os.path.join(save_dir, f'Brightness_{i}_{base_name}.jpg'), brightness_image)

    # Contrast
    for i in range(1, 3):
        contrast_image = cv2.convertScaleAbs(img_np, alpha=i, beta=0)
        cv2.imwrite(os.path.join(save_dir, f'Contrast_{i}_{base_name}.jpg'), contrast_image)

    # Noise
    noise_image = img_np.copy()
    noise_mask = np.random.randint(0, 100, noise_image.shape).astype('uint8')
    noise_image = cv2.add(noise_image, noise_mask)
    cv2.imwrite(os.path.join(save_dir, f'Noise_{base_name}.jpg'), noise_image)

def augment_dataset(folder_path):
    save_dir = os.path.join(folder_path, "Augmented Images")
    os.makedirs(save_dir, exist_ok=True)
    
    for file_name in os.listdir(folder_path):
        if file_name == "Augmented Images":
            continue
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, file_name)
            base_name = os.path.basename(image_path)
            while '.' in base_name:
                base_name = os.path.splitext(base_name)[0]
            image_save_dir = os.path.join(save_dir, base_name)
            os.makedirs(image_save_dir, exist_ok=True)
            augment_image(image_path, image_save_dir)
    
    print(f"Augmention Saved At {save_dir}")

def main():
    current_folder = os.path.dirname(os.path.realpath(__file__))
    augment_dataset(current_folder)

if __name__ == "__main__":
    main()
