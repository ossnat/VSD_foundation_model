# preprocessing/steps.py
from src.preprocessing.preprocess_pipeline import PreprocessingStep

import cv2
import numpy as np
from scipy import ndimage
from skimage import morphology, measure, filters
from typing import Tuple, Optional, Dict, Any
import logging


class ChamberCropper(PreprocessingStep):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)

    def process(self, data: np.ndarray) -> np.ndarray:
        """
        Process VSD data to crop to chamber region
        Args:
            data: Shape [trials, frames, height, width] or [frames, height, width]
        Returns:
            Cropped data maintaining original structure
        """
        # Handle different input shapes
        original_shape = data.shape
        if len(data.shape) == 4:  # [trials, frames, H, W]
            # Average across trials and frames for stable ROI detection
            detection_image = np.mean(data, axis=(0, 1))
        elif len(data.shape) == 3:  # [frames, H, W]
            # Average across frames
            detection_image = np.mean(data, axis=0)
        else:
            raise ValueError(f"Unexpected data shape: {data.shape}")

        # Detect chamber ROI using the averaged image
        roi_mask, bbox = self._detect_chamber_roi(detection_image)

        # Apply cropping to original data
        cropped_data = self._apply_crop(data, bbox)

        # Save mask for inspection if configured
        if self.config.get('save_masks', False):
            self._save_roi_mask(roi_mask, bbox)

        return cropped_data

    def _detect_chamber_roi(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Detect chamber region in VSD image using multiple robust methods
        Args:
            image: 2D averaged intensity image [H, W]
        Returns:
            roi_mask: Binary mask of chamber region
            bbox: Bounding box (x, y, width, height)
        """
        method = self.config.get('detection_method', 'adaptive_threshold')

        if method == 'adaptive_threshold':
            return self._adaptive_threshold_detection(image)
        elif method == 'edge_detection':
            return self._edge_based_detection(image)
        elif method == 'hybrid':
            return self._hybrid_detection(image)
        else:
            raise ValueError(f"Unknown detection method: {method}")

    def _adaptive_threshold_detection(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Adaptive thresholding approach robust to intensity variations
        """
        # Normalize image to 0-255 range for consistent processing
        img_norm = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

        # Apply Gaussian blur to reduce noise
        blur_sigma = self.config.get('blur_sigma', 2.0)
        img_blur = cv2.GaussianBlur(img_norm, (0, 0), blur_sigma)

        # Adaptive thresholding - automatically finds optimal threshold
        # Using Otsu's method which is robust across different intensity distributions
        _, binary_mask = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Alternative: Use local adaptive thresholding if global doesn't work well
        if binary_mask.sum() < 0.05 * binary_mask.size or binary_mask.sum() > 0.8 * binary_mask.size:
            self.logger.warning("Global thresholding failed, trying adaptive thresholding")
            binary_mask = cv2.adaptiveThreshold(
                img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 51, 2
            )

        # Clean up the mask using morphological operations
        roi_mask = self._clean_binary_mask(binary_mask)

        # Extract bounding box
        bbox = self._get_largest_component_bbox(roi_mask)

        return roi_mask, bbox

    def _edge_based_detection(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Edge-based circular detection using Hough transforms
        """
        # Normalize and convert to uint8
        img_norm = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

        # Apply median filter to reduce noise while preserving edges
        img_filtered = cv2.medianBlur(img_norm, 5)

        # Detect circles using Hough Circle Transform
        min_radius = self.config.get('min_radius', 20)
        max_radius = self.config.get('max_radius', min(image.shape) // 2)

        circles = cv2.HoughCircles(
            img_filtered,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=max_radius // 2,
            param1=100,
            param2=30,
            minRadius=min_radius,
            maxRadius=max_radius
        )

        if circles is not None:
            # Use the largest detected circle
            circles = np.round(circles[0, :]).astype("int")
            largest_circle = max(circles, key=lambda c: c[2])  # Select by radius

            # Create circular mask
            roi_mask = np.zeros(image.shape, dtype=np.uint8)
            cv2.circle(roi_mask, (largest_circle[0], largest_circle[1]), largest_circle[2], 255, -1)

            # Create bounding box around circle
            x, y, r = largest_circle
            bbox = (
            max(0, x - r), max(0, y - r), min(2 * r, image.shape[1] - (x - r)), min(2 * r, image.shape[0] - (y - r)))
        else:
            # Fallback to adaptive thresholding if no circles detected
            self.logger.warning("No circles detected, falling back to adaptive thresholding")
            return self._adaptive_threshold_detection(image)

        return roi_mask, bbox

    def _hybrid_detection(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Combine adaptive thresholding with geometric constraints
        """
        # First try adaptive thresholding
        threshold_mask, threshold_bbox = self._adaptive_threshold_detection(image)

        # Then try edge detection
        try:
            edge_mask, edge_bbox = self._edge_based_detection(image)

            # Combine masks using intersection (more conservative)
            combined_mask = cv2.bitwise_and(threshold_mask, edge_mask)

            # If intersection is too small, use the larger mask
            if combined_mask.sum() < 0.1 * max(threshold_mask.sum(), edge_mask.sum()):
                if threshold_mask.sum() > edge_mask.sum():
                    return threshold_mask, threshold_bbox
                else:
                    return edge_mask, edge_bbox
            else:
                bbox = self._get_largest_component_bbox(combined_mask)
                return combined_mask, bbox

        except Exception as e:
            self.logger.warning(f"Edge detection failed: {e}, using threshold mask")
            return threshold_mask, threshold_bbox

    def _clean_binary_mask(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Clean binary mask using morphological operations
        """
        # Convert to boolean for skimage operations
        mask_bool = binary_mask > 0

        # Remove small objects (noise)
        min_size = self.config.get('min_region_size', 500)
        mask_cleaned = morphology.remove_small_objects(mask_bool, min_size=min_size)

        # Fill holes in the mask
        mask_filled = ndimage.binary_fill_holes(mask_cleaned)

        # Apply morphological opening to separate connected regions
        kernel_size = self.config.get('morphology_kernel_size', 3)
        kernel = morphology.disk(kernel_size)
        mask_opened = morphology.opening(mask_filled, kernel)

        # Convert back to uint8
        return (mask_opened * 255).astype(np.uint8)

    def _get_largest_component_bbox(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Get bounding box of largest connected component
        """
        # Find connected components
        labeled_mask = measure.label(mask > 0)
        regions = measure.regionprops(labeled_mask)

        if not regions:
            # If no regions found, return full image bbox
            h, w = mask.shape
            return (0, 0, w, h)

        # Get largest region by area
        largest_region = max(regions, key=lambda r: r.area)

        # Extract bounding box with padding
        min_row, min_col, max_row, max_col = largest_region.bbox
        padding = self.config.get('padding_pixels', 5)

        # Apply padding while staying within image bounds
        h, w = mask.shape
        x = max(0, min_col - padding)
        y = max(0, min_row - padding)
        width = min(w - x, max_col - min_col + 2 * padding)
        height = min(h - y, max_row - min_row + 2 * padding)

        return (x, y, width, height)

    def _apply_crop(self, data: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Apply bounding box crop to data while preserving structure
        """
        x, y, width, height = bbox

        if len(data.shape) == 4:  # [trials, frames, H, W]
            return data[:, :, y:y + height, x:x + width]
        elif len(data.shape) == 3:  # [frames, H, W]
            return data[:, y:y + height, x:x + width]
        else:
            raise ValueError(f"Unexpected data shape: {data.shape}")

    def _save_roi_mask(self, mask: np.ndarray, bbox: Tuple[int, int, int, int]):
        """
        Save ROI mask and bbox for inspection
        """
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # Plot original mask
        ax1.imshow(mask, cmap='gray')
        ax1.set_title('Detected Chamber ROI')
        ax1.axis('off')

        # Plot bounding box overlay
        x, y, w, h = bbox
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
        ax2.imshow(mask, cmap='gray')
        ax2.add_patch(rect)
        ax2.set_title('Bounding Box')
        ax2.axis('off')

        plt.tight_layout()
        plt.savefig('roi_detection_debug.png', dpi=150, bbox_inches='tight')
        plt.close()

    def get_params(self) -> Dict[str, Any]:
        """Return step parameters for logging/reproducibility"""
        return {
            'detection_method': self.config.get('detection_method', 'adaptive_threshold'),
            'blur_sigma': self.config.get('blur_sigma', 2.0),
            'min_region_size': self.config.get('min_region_size', 500),
            'padding_pixels': self.config.get('padding_pixels', 5),
            'morphology_kernel_size': self.config.get('morphology_kernel_size', 3)
        }

    def _simple_oval_detection(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Simple method to detect oval/circular chamber and zero out edges
        Preserves original pixel values inside detected contour, zeros outside
        """
        h, w = image.shape

        # Method 1: Try to detect actual oval/circular contours
        detected_contour = self._detect_actual_contour(image)

        if detected_contour is not None:
            # Use detected contour
            mask = self._create_contour_mask(detected_contour, image.shape)
            coverage = np.sum(mask) / (h * w)

            # Check if contour covers reasonable portion of image (at least 50%)
            if coverage >= 0.5:
                # Apply mask: keep original values inside, zero outside
                result_image = image * mask

                # Create bounding box of the contour for consistency
                bbox = self._contour_to_bbox(detected_contour, image.shape)

                return (mask * 255).astype(np.uint8), bbox

        # Method 2: Fallback to largest inscribed circle
        return self._largest_inscribed_circle(image)

    def _detect_actual_contour(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Try to detect actual oval/circular contours in the VSD image
        Uses edge detection on enhanced contrast version
        """
        # Enhance contrast to make chamber edges more visible
        # Normalize to 0-255 for consistent processing
        img_norm = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance edges
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img_enhanced = clahe.apply(img_norm)

        # Apply Gaussian blur to reduce noise while preserving edges
        img_blur = cv2.GaussianBlur(img_enhanced, (5, 5), 1.0)

        # Use Canny edge detection with conservative thresholds
        edges = cv2.Canny(img_blur, 30, 100)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Filter contours by area and shape
        valid_contours = []
        h, w = image.shape
        min_area = 0.3 * h * w  # At least 30% of image
        max_area = 0.9 * h * w  # At most 90% of image

        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                # Check if contour is roughly circular/oval
                # Fit ellipse and check aspect ratio
                if len(contour) >= 5:  # Need at least 5 points to fit ellipse
                    try:
                        ellipse = cv2.fitEllipse(contour)
                        axes = ellipse[1]  # (major_axis, minor_axis)
                        aspect_ratio = max(axes) / min(axes)

                        # Accept if reasonably circular/oval (aspect ratio < 2.5)
                        if aspect_ratio <= 2.5:
                            valid_contours.append((contour, area))
                    except:
                        continue

        if not valid_contours:
            return None

        # Return largest valid contour
        largest_contour = max(valid_contours, key=lambda x: x[1])[0]
        return largest_contour

    def _create_contour_mask(self, contour: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        """Create binary mask from contour"""
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(mask, [contour], 1)
        return mask

    def _contour_to_bbox(self, contour: np.ndarray, shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Convert contour to bounding box"""
        x, y, w, h = cv2.boundingRect(contour)
        # Add small padding
        padding = 5
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(shape[1] - x, w + 2 * padding)
        h = min(shape[0] - y, h + 2 * padding)
        return (x, y, w, h)

    def _largest_inscribed_circle(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Fallback method: create largest circle that fits in 100x100 image
        Centers the circle and ensures most of image is preserved
        """
        h, w = image.shape

        # Find center of image
        center_x, center_y = w // 2, h // 2

        # Calculate largest radius that fits in the image with some margin
        # Use 85% of the distance to nearest edge to ensure we don't clip
        margin_factor = 0.85
        radius = int(min(center_x, center_y, w - center_x, h - center_y) * margin_factor)

        # Ensure minimum reasonable radius
        min_radius = min(h, w) // 4
        radius = max(radius, min_radius)

        # Create circular mask
        y_coords, x_coords = np.ogrid[:h, :w]
        mask = ((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2) <= radius ** 2

        # Apply mask: preserve original values inside circle, zero outside
        result_image = image * mask

        # Create bounding box around circle
        x = max(0, center_x - radius)
        y = max(0, center_y - radius)
        bbox_w = min(2 * radius, w - x)
        bbox_h = min(2 * radius, h - y)
        bbox = (x, y, bbox_w, bbox_h)

        return (mask * 255).astype(np.uint8), bbox
