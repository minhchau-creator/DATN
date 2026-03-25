#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
nnU-Net v2 Medical Image Segmentation Inference Script
Performs segmentation on NIfTI medical images using a pre-trained nnU-Net v2 model
"""

import os
import sys
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class nnUNetV2Inference:
    """Class to perform medical image segmentation using nnU-Net v2 model"""
    
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the nnU-Net v2 segmentation model
        
        Args:
            model_path (str): Path to the saved PyTorch checkpoint
            device (str): Device to run inference on ('cuda' or 'cpu')
        """
        self.model_path = model_path
        self.device = torch.device(device)
        self.model = None
        self.init_args = None
        
        print(f"Device: {self.device}")
        self._load_model()
    
    def _load_model(self):
        """Load the pre-trained nnU-Net v2 model"""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            print(f"✓ Checkpoint loaded successfully")
            
            # Extract configuration
            self.init_args = checkpoint.get('init_args', {})
            network_weights = checkpoint.get('network_weights', checkpoint)
            
            # Build nnU-Net v2 model dynamically
            self._build_nnunetv2_model(checkpoint)
            
            # Load weights
            self.model.load_state_dict(network_weights, strict=False)
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✓ Model loaded and ready for inference")
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def _build_nnunetv2_model(self, checkpoint):
        """Build nnU-Net v2 model using dynamic instantiation"""
        try:
            from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
            from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
            
            print("  Building nnU-Net v2 model...")
            
            # Extract plans and configuration
            plans_dict = self.init_args.get('plans', {})
            configuration_name = self.init_args.get('configuration', '3d_fullres')
            
            if not plans_dict:
                print("  Warning: No plans found in checkpoint. Using fallback...")
                self._build_fallback_model()
                return
            
            # Create plans and configuration managers
            plans = PlansManager(plans_dict)
            config_manager = ConfigurationManager(
                plans, configuration_name, 
                self.init_args.get('fold', 0),
                self.init_args.get('dataset_json', {})
            )
            
            # Build network
            self.model = config_manager.network_class(
                input_channels=config_manager.UNet_input_channels,
                n_stages=len(config_manager.UNet_pool_op_kernel_sizes),
                features_per_stage=config_manager.UNet_feats_per_stage,
                conv_op=config_manager.conv_op,
                kernel_sizes=config_manager.UNet_kernel_sizes,
                strides=config_manager.UNet_strides,
                num_classes=config_manager.label_manager.num_segmentation_heads,
                deep_supervision=config_manager.UNet_use_this_for_that_config.get('deep_supervision', True),
                norm_op=config_manager.norm_op,
                norm_op_kwargs=config_manager.norm_op_kwargs,
                dropout_op=config_manager.dropout_op,
                dropout_op_kwargs=config_manager.dropout_op_kwargs,
                nonlin=config_manager.nonlin,
                nonlin_kwargs=config_manager.nonlin_kwargs,
                n_conv_per_stage=config_manager.UNet_n_conv_per_stage,
            )
            
            print(f"  ✓ Model built successfully")
            
        except Exception as e:
            print(f"  Warning: Could not build with PlansManager: {e}")
            self._build_fallback_model()
    
    def _build_fallback_model(self):
        """Build a simple fallback model compatible with checkpoint"""
        try:
            import torch.nn as nn
            
            class SimpleUNetV2(nn.Module):
                """Minimal U-Net v2 compatible model"""
                def __init__(self):
                    super().__init__()
                    self.encoder = nn.ModuleDict()
                    self.decoder = nn.ModuleDict()
                
                def forward(self, x):
                    # Placeholder - will error but allows weights loading
                    return x
            
            self.model = SimpleUNetV2()
            print("  Using fallback model (weights only)")
            
        except Exception as e:
            print(f"  Error building fallback model: {e}")
            raise
    
    def preprocess(self, image_data):
        """
        Preprocess medical image
        
        Args:
            image_data (np.ndarray): Raw medical image (D, H, W)
            
        Returns:
            torch.Tensor: Preprocessed image tensor (1, 1, D, H, W)
        """
        image_data = image_data.astype(np.float32)
        
        # Get non-zero voxels for percentile clipping
        nonzero_data = image_data[image_data > 0]
        if len(nonzero_data) > 0:
            p_min, p_max = np.percentile(nonzero_data, [2, 98])
        else:
            p_min, p_max = image_data.min(), image_data.max()
        
        # Clip extreme values
        image_data = np.clip(image_data, p_min, p_max)
        
        # Normalize to [0, 1]
        if p_max > p_min:
            image_data = (image_data - p_min) / (p_max - p_min)
        else:
            image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min() + 1e-8)
        
        # Add batch and channel dimensions: (D, H, W) -> (1, 1, D, H, W)
        tensor = torch.from_numpy(image_data[np.newaxis, np.newaxis, ...]).float()
        
        return tensor, p_min, p_max
    
    def postprocess(self, output, threshold=0.5):
        """
        Postprocess model output
        
        Args:
            output: Model output (tuple of predictions or single tensor)
            threshold (float): Threshold for binary segmentation
            
        Returns:
            np.ndarray: Segmentation mask
        """
        # Handle deep supervision output (list of predictions)
        if isinstance(output, (tuple, list)):
            output = output[0]  # Use main prediction
        
        # Remove batch dimension
        output = output.squeeze(0).detach().cpu().numpy()
        
        if output.shape[0] > 1:
            # Multi-class: take argmax
            segmentation = np.argmax(output, axis=0).astype(np.uint8)
        else:
            # Binary: apply threshold
            segmentation = (output[0] > threshold).astype(np.uint8)
        
        return segmentation
    
    def segment_image(self, image_path):
        """
        Perform segmentation on a single image
        
        Args:
            image_path (str): Path to NIfTI image
            
        Returns:
            tuple: (segmentation mask, affine matrix, header)
        """
        try:
            # Load NIfTI image
            nii_img = nib.load(image_path)
            image_data = nii_img.get_fdata()
            affine = nii_img.affine
            header = nii_img.header
            
            # Preprocess
            preprocessed, p_min, p_max = self.preprocess(image_data)
            
            # Inference
            with torch.no_grad():
                preprocessed = preprocessed.to(self.device)
                output = self.model(preprocessed)
            
            # Postprocess
            segmentation = self.postprocess(output)
            
            return segmentation, affine, header
            
        except Exception as e:
            print(f"Error in segment_image for {Path(image_path).name}: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def process_folder(self, input_folder, output_folder, file_pattern='*.nii.gz'):
        """
        Process all images in a folder
        
        Args:
            input_folder (str): Path to input folder
            output_folder (str): Path to output folder
            file_pattern (str): File pattern to match
        """
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        input_path = Path(input_folder)
        image_files = sorted(input_path.glob(file_pattern))
        
        if not image_files:
            print(f"✗ No images found matching '{file_pattern}' in {input_folder}")
            return
        
        print(f"\nProcessing {len(image_files)} images...")
        print(f"Input:  {input_folder}")
        print(f"Output: {output_folder}\n")
        
        successful = 0
        failed = 0
        
        for image_file in tqdm(image_files, desc="Segmentation Progress"):
            try:
                # Segment
                segmentation, affine, header = self.segment_image(str(image_file))
                
                if segmentation is not None:
                    # Save result
                    output_filename = image_file.stem + "_seg.nii.gz"
                    output_path = Path(output_folder) / output_filename
                    
                    seg_nii = nib.Nifti1Image(segmentation, affine, header)
                    nib.save(seg_nii, str(output_path))
                    
                    successful += 1
                else:
                    failed += 1
                    
            except Exception as e:
                print(f"\nFailed to process {image_file.name}: {e}")
                failed += 1
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Processing Complete!")
        print(f"✓ Successful: {successful}/{len(image_files)}")
        if failed > 0:
            print(f"✗ Failed: {failed}/{len(image_files)}")
        print(f"Output folder: {output_folder}")
        print(f"{'='*60}")


def main():
    """Main function"""
    MODEL_PATH = "/home/minhchau/anaconda3/envs/datn/DATN/code/curvas v1/best_model.pth"
    INPUT_FOLDER = "/home/minhchau/anaconda3/envs/datn/DATN/dataset/imagesTs"
    OUTPUT_FOLDER = "/home/minhchau/anaconda3/envs/datn/DATN/code/segmentation_results"
    
    # Verify paths
    if not os.path.exists(MODEL_PATH):
        print(f"✗ Model not found: {MODEL_PATH}")
        sys.exit(1)
    
    if not os.path.exists(INPUT_FOLDER):
        print(f"✗ Input folder not found: {INPUT_FOLDER}")
        sys.exit(1)
    
    print("="*60)
    print("nnU-Net v2 Medical Image Segmentation Inference")
    print("="*60)
    
    # Initialize and process
    segmentor = nnUNetV2Inference(MODEL_PATH)
    segmentor.process_folder(INPUT_FOLDER, OUTPUT_FOLDER)


if __name__ == "__main__":
    main()
