#!/usr/bin/env python3
"""
Example script to process a multi-timepoint, two-channel .czi or .tiff file to analyze tube-like structures.
It applies:
  - A filtering step (either median filtering or Gaussian filtering on both channels),
  - (Optional) Background subtraction using a white top-hat transform,
  - A tubeness filter (using the Frangi filter, with parameters from parameters.txt),
  - Segmentation via thresholding,
  - Removal of small segmented objects (smaller than a given area),
  - Labeling of each segmented region,
  - Intensity analysis (mean & median) and area per labeled region,
  - Creation of overlay images (segmentation mask over the original).
All outputs are saved as single multi-timepoint files.
Parameters (filter sizes, I/O directories, Frangi parameters, etc.) are read from "parameters.txt".
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import configparser
import tifffile as tiff
from scipy.ndimage import median_filter
from skimage import morphology, exposure
from skimage.filters import frangi, threshold_otsu
from skimage.morphology import disk, remove_small_objects
from skimage.measure import label, regionprops
from aicsimageio import AICSImage

# --------------------------
# Utility Functions
# --------------------------

def load_parameters(param_file):
    with open(param_file, 'r', encoding='utf-8-sig') as f:
        content = f.read()
    print("File content:", content)
    config_obj = configparser.ConfigParser()
    config_obj.read_string(content)
    print("Loaded sections:", config_obj.sections())

    base_dir = os.path.dirname(os.path.abspath(__file__))
    params = {
        'input_dir': os.path.abspath(os.path.join(base_dir, config_obj.get('input', 'input_dir'))),
        'output_dir': os.path.abspath(os.path.join(base_dir, config_obj.get('input', 'output_dir'))),
        'filter_type': config_obj.get('filter', 'filter_type'),
        'median_filter_size': config_obj.getint('filter', 'median_filter_size'),
        'gaussian_sigma': config_obj.getfloat('filter', 'gaussian_sigma'),
        'rolling_ball_radius': config_obj.getint('filter', 'rolling_ball_radius'),
        'tubeness_scale': config_obj.getfloat('filter', 'tubeness_scale'),
        'threshold_factor': config_obj.getfloat('filter', 'threshold_factor'),
        'apply_background_subtraction': config_obj.getboolean('filter', 'apply_background_subtraction'),
        'min_region_area': config_obj.getint('filter', 'min_region_area'),
        'frangi_alpha': config_obj.getfloat('frangi', 'alpha'),
        'frangi_beta': config_obj.getfloat('frangi', 'beta'),
        'frangi_gamma': config_obj.getfloat('frangi', 'gamma'),
        'frangi_sigmas': [float(s.strip()) for s in config_obj.get('frangi', 'sigmas').split(',')]
    }
    return params

def create_out_dirs(base_out_dir, steps):
    out_dirs = {}
    for step in steps:
        step_dir = os.path.join(base_out_dir, step)
        os.makedirs(step_dir, exist_ok=True)
        out_dirs[step] = step_dir
    return out_dirs

def subtract_background(image, radius):
    selem = disk(radius)
    bg_subtracted = morphology.white_tophat(image, selem)
    return bg_subtracted

def create_overlay(original, mask, alpha=0.3):
    norm_orig = exposure.rescale_intensity(original, out_range=(0, 1))
    rgb_orig = np.dstack([norm_orig]*3)
    overlay = rgb_orig.copy()
    overlay[..., 0] = np.where(mask, 1, overlay[..., 0])
    blended = (1 - alpha) * rgb_orig + alpha * overlay
    return blended

def process_image(image, params):
    """
    Process a 2D image with:
      - Filtering (either median or Gaussian),
      - (Optional) Background subtraction,
      - Tubeness filtering (Frangi),
      - Segmentation via thresholding.
    Returns a dictionary with intermediate results.
    """
    results = {}
    # Choose filtering type based on parameter
    if params['filter_type'].lower() == 'gaussian':
        from scipy.ndimage import gaussian_filter
        filtered = gaussian_filter(image, sigma=params['gaussian_sigma'])
    else:
        filtered = median_filter(image, size=params['median_filter_size'])
    results['median'] = filtered  # Note: still storing under 'median' key for consistency

    # Optional background subtraction
    if params['apply_background_subtraction']:
        bg_subtracted = subtract_background(filtered, radius=params['rolling_ball_radius'])
        bg_subtracted = exposure.rescale_intensity(bg_subtracted)
    else:
        bg_subtracted = filtered
    results['background_subtracted'] = bg_subtracted

    # Tubeness via Frangi filter
    tubeness = frangi(bg_subtracted, 
                      sigmas=params['frangi_sigmas'],
                      alpha=params['frangi_alpha'],
                      beta=params['frangi_beta'],
                      gamma=params['frangi_gamma'],
                      black_ridges=False)
    results['tubeness'] = tubeness

    # Segmentation: Otsu threshold adjusted by threshold_factor
    otsu_thresh = threshold_otsu(tubeness)
    seg_thresh = otsu_thresh * params['threshold_factor']
    segmented = tubeness > seg_thresh
    results['segmented'] = segmented

    return results

def analyze_region_properties(original, labeled_mask):
    """
    Compute region properties for each labeled region.
    Returns a list of dictionaries containing label, area, mean intensity, and median intensity.
    """
    regions = regionprops(labeled_mask, intensity_image=original)
    props_list = []
    for region in regions:
        median_intensity = np.median(region.intensity_image[region.image])
        props = {
            'region_label': region.label,
            'area': region.area,
            'mean_intensity': region.mean_intensity,
            'median_intensity': median_intensity
        }
        props_list.append(props)
    return props_list

# --------------------------
# Main Processing
# --------------------------

def main():
    param_file = os.path.join(os.path.dirname(__file__), "..", "parameters.txt")
    params = load_parameters(param_file)
    
    # Create output directories for each processing step.
    steps = ['median_filtered', 'background_subtracted', 'tubeness', 'segmented', 'overlays']
    out_dirs = create_out_dirs(params['output_dir'], steps)
    # Additional directories for combined masks and overlays:
    combined_seg_dir = os.path.join(params['output_dir'], 'combined_segmented')
    combined_overlay_dir = os.path.join(params['output_dir'], 'combined_overlays')
    os.makedirs(combined_seg_dir, exist_ok=True)
    os.makedirs(combined_overlay_dir, exist_ok=True)
    
    input_files = [f for f in os.listdir(params['input_dir']) if f.lower().endswith(('.czi', '.tiff', '.tif'))]
    if not input_files:
        print("No .czi or .tiff files found in the input directory.")
        return
    
    intensity_results = []
    
    for file_name in input_files:
        file_path = os.path.join(params['input_dir'], file_name)
        print(f"Processing file: {file_path}")
        
        img_obj = AICSImage(file_path)
        image_data = img_obj.get_image_data("TCYX")  # shape: (T, C, H, W)
        T, C, H, W = image_data.shape
        
        # Initialize arrays to accumulate results over time and channels.
        med_filtered_arr = np.zeros((T, C, H, W), dtype=np.float32)
        bg_subtracted_arr = np.zeros((T, C, H, W), dtype=np.float32)
        tubeness_arr = np.zeros((T, C, H, W), dtype=np.float32)
        segmented_arr = np.zeros((T, C, H, W), dtype=bool)
        overlays_arr = np.zeros((T, C, H, W, 3), dtype=np.uint8)
        originals_arr = np.zeros((T, C, H, W), dtype=np.float32)
        
        # Process each timepoint and channel.
        for t in range(T):
            for c in range(C):
                original = image_data[t, c, :, :].astype(np.float32)
                originals_arr[t, c, :, :] = original
                proc_results = process_image(original, params)
                med_filtered_arr[t, c, :, :] = proc_results['median']
                bg_subtracted_arr[t, c, :, :] = proc_results['background_subtracted']
                tubeness_arr[t, c, :, :] = proc_results['tubeness']
                segmented_arr[t, c, :, :] = proc_results['segmented']
                overlay = create_overlay(original, proc_results['segmented'])
                overlays_arr[t, c, :, :, :] = (overlay * 255).astype(np.uint8)
        
        # Compute combined segmentation mask (logical AND across channels) per timepoint.
        combined_mask_arr = np.zeros((T, H, W), dtype=bool)
        for t in range(T):
            combined_mask = segmented_arr[t, 0, :, :]
            for c in range(1, C):
                combined_mask = np.logical_and(combined_mask, segmented_arr[t, c, :, :])
            # Remove small objects
            filtered_mask = remove_small_objects(combined_mask, min_size=params['min_region_area'])
            combined_mask_arr[t, :, :] = filtered_mask
            
            # For each channel, analyze region properties using the combined mask.
            labeled_mask = label(filtered_mask)
            for c in range(C):
                region_props = analyze_region_properties(originals_arr[t, c, :, :], labeled_mask)
                for props in region_props:
                    record = {
                        'file': file_name,
                        'timepoint': t,
                        'channel': c,
                        'region_label': props['region_label'],
                        'area': props['area'],
                        'mean_intensity': props['mean_intensity'],
                        'median_intensity': props['median_intensity'],
                        'mask': 'combined'
                    }
                    intensity_results.append(record)
        
        # Save multi-timepoint outputs as single files per processing step.
        base_out_name = os.path.splitext(file_name)[0]
        tiff.imwrite(os.path.join(out_dirs['median_filtered'], f"{base_out_name}_median_filtered.tif"),
                     med_filtered_arr)
        tiff.imwrite(os.path.join(out_dirs['background_subtracted'], f"{base_out_name}_background_subtracted.tif"),
                     bg_subtracted_arr)
        tiff.imwrite(os.path.join(out_dirs['tubeness'], f"{base_out_name}_tubeness.tif"),
                     tubeness_arr)
        # Save segmented masks as uint8 (0/1)
        tiff.imwrite(os.path.join(out_dirs['segmented'], f"{base_out_name}_segmented.tif"),
                     segmented_arr.astype(np.uint8))
        tiff.imwrite(os.path.join(out_dirs['overlays'], f"{base_out_name}_overlays.tif"),
                     overlays_arr)
        tiff.imwrite(os.path.join(combined_seg_dir, f"{base_out_name}_combined_filtered.tif"),
                     combined_mask_arr.astype(np.uint8))
        
        # For combined overlays, save one multi-timepoint file per channel.
        for c in range(C):
            combined_overlay_arr = np.zeros((T, H, W, 3), dtype=np.uint8)
            for t in range(T):
                combined_overlay = create_overlay(originals_arr[t, c, :, :], combined_mask_arr[t, :, :])
                combined_overlay_arr[t, :, :, :] = (combined_overlay * 255).astype(np.uint8)
            tiff.imwrite(os.path.join(combined_overlay_dir, f"{base_out_name}_c{c:02d}_combined_overlay.tif"),
                         combined_overlay_arr)
    
    df = pd.DataFrame(intensity_results)
    csv_path = os.path.join(params['output_dir'], "intensity_analysis.csv")
    df.to_csv(csv_path, index=False)
    print(f"Intensity analysis saved to {csv_path}")

if __name__ == "__main__":
    main()