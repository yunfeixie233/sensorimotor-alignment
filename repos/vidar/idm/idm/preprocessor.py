import torch
import numpy as np
import torchvision
from PIL import Image


class DinoPreprocessor:
    """Handles image preprocessing."""

    AUG_CONFIG = {
        'brightness_range': ((0.8, 1.2), (0.4, 1.6)),
        'contrast_range': ((0.7, 1.3), (0.7, 1.7)),
        'saturation_range': ((0.5, 1.5), (0.5, 2.0)),
        'hue_shift': (0.05, 0.10),
        'random_apply_prob': (0.4, 0.8),
        'sharpness_factor': (1.8, 2.2),
        'sharpness_prob': (0.7, 0.9)
    }
    
    def __init__(self, args):
        self.use_transform = args.use_transform
        self.current_progress = 0.0
        
        self.init_transforms()
    
    def init_transforms(self):
        # Determine image dimensions based on camera configuration
        self.height = 720  # 480 + 240
        self.width = 640
        self.dino_size = 518
        
        self.transform = self._build_transform()
        
        # DINO normalization
        self.normalize_dino = torchvision.transforms.Compose([
            torchvision.transforms.Resize((self.dino_size, self.dino_size)),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def _lerp(self, start, end, progress):
        return start + (end - start) * progress

    def _build_transform(self):
        p = self.current_progress
        
        params = {}
        for key, (start, end) in self.AUG_CONFIG.items():
            if isinstance(start, tuple):
                params[key] = (
                    self._lerp(start[0], end[0], p),
                    self._lerp(start[1], end[1], p)
                )
            else:
                params[key] = self._lerp(start, end, p)
        
        return torchvision.transforms.Compose([
            torchvision.transforms.Resize((self.height, self.width)),
            torchvision.transforms.ColorJitter(
                brightness=params['brightness_range'],
                contrast=params['contrast_range'],
                saturation=params['saturation_range'],
                hue=params['hue_shift']
            ),
        ])

    def set_augmentation_progress(self, progress):
        self.current_progress = max(0.0, min(1.0, progress))
        self.transform = self._build_transform()
    
    def process_image(self, image):
        """Process a single image for model input. input shape is [H, W, C]"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        if isinstance(image, Image.Image):
            if self.use_transform:
                image = self.transform(image)
            image = torchvision.transforms.functional.to_tensor(image).float()

        return self.normalize_dino(image)
    
    def process_batch(self, images, pos=None):
        """Process a batch of images and optionally handle position data for DINO model."""
        # For standard mode, just process the RGB images
        processed_images = torch.stack([self.process_image(img) for img in images])
        
        return processed_images
    
    def handle_flip(self, images, pos):
        """Handle flipping of images and position data. FOR SINGLE IMAGE AND POS!"""
        if isinstance(images, Image.Image):
            flipped_images = images.transpose(Image.FLIP_LEFT_RIGHT)
        
        elif isinstance(images, torch.Tensor):
            # Flip images horizontally
            flipped_images = torch.flip(images, [2])
        
        # Create flipped position vector
        flipped_pos = torch.zeros_like(pos)
        flipped_pos[:7] = pos[7:]
        flipped_pos[7:] = pos[:7]
        
        # Negate specific components that need to be reversed
        flipped_pos[[0, 4, 5, 7, 11, 12]] *= -1
        
        return flipped_images, flipped_pos
    
    def flip_images_pos_batch(self, images, pos):
        """
        randomly flip images and positions, images shape is [B, 3, H, W], pos shape is [B, 14]
        """
        flip_flags = torch.randint(0, 2, (len(images), 1))
        # Handle flipping if needed
        if flip_flags.any():
            # Only flip the images and positions where flip_flag is True
            for i, flip in enumerate(flip_flags):
                if flip:
                    images[i], pos[i] = self.handle_flip(images[i], pos[i])
        return images, pos
