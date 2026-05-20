import cv2
import numpy as np

class Preprocessor():
    def __init__(self):
        pass


    def resize(self, img, patch_image_size=480):
        height, width = img.shape[:2]

        if width >= height:
            new_width = max(64, patch_image_size)
            new_height = max(64, int(patch_image_size * (height / width)))
            top = (patch_image_size - new_height) // 2
            bottom = patch_image_size - new_height - top
            left, right = 0, 0
        else:
            new_height = max(64, patch_image_size)
            new_width = max(64, int(patch_image_size * (width / height)))
            left = (patch_image_size - new_width) // 2
            right = patch_image_size - new_width - left
            top, bottom = 0, 0

        # Resize image
        img_new = cv2.resize(img, (new_width, new_height))

        # Pad image
        img_new = cv2.copyMakeBorder(img_new, top, bottom, left, right, borderType=cv2.BORDER_REPLICATE)

        assert img_new.shape[0] == patch_image_size and img_new.shape[1] == patch_image_size

        return img_new

    def normalize(self, img):
        # Convert to float32 and normalize
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
        img = (img - mean) / std
        return img

    def __call__(self, input):
        image = cv2.imread(input)  # 默认读取为BGR
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 转换为 RGB 格式


        patch_image = self.resize(np.array(image))
        patch_image = patch_image.transpose((2, 0, 1))  # HWC -> CHW
        patch_image = self.normalize(patch_image)
        patch_image = np.expand_dims(patch_image, axis=0)
        return patch_image

