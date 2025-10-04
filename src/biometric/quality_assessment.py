"""
Image quality assessment module for biometric enrollment and authentication.

This module evaluates face image quality to ensure optimal biometric
performance and reject poor quality samples that could affect accuracy.
"""

import cv2
import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from ..core.logging import get_logger, PerformanceTimer

logger = get_logger(__name__)


class QualityMetric(Enum):
    """Types of quality metrics."""
    SHARPNESS = "sharpness"
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"
    FACE_SIZE = "face_size"
    POSE_ANGLE = "pose_angle"
    EYE_OPENNESS = "eye_openness"
    MOTION_BLUR = "motion_blur"
    COMPRESSION_ARTIFACTS = "compression_artifacts"


@dataclass
class QualityThresholds:
    """Thresholds for quality assessment."""
    min_sharpness: float = 100.0
    max_sharpness: float = 2000.0
    min_brightness: float = 80.0
    max_brightness: float = 200.0
    min_contrast: float = 30.0
    min_face_size: int = 100
    max_pose_angle: float = 30.0  # degrees
    min_eye_openness: float = 0.3
    max_motion_blur: float = 0.8


@dataclass
class QualityResult:
    """Result of quality assessment."""
    overall_score: float
    quality_metrics: Dict[str, float]
    passed_checks: Dict[str, bool]
    recommendations: List[str]


class QualityAssessment:
    """
    Comprehensive image quality assessment for biometric applications.

    Evaluates multiple quality factors:
    - Image sharpness and focus
    - Lighting conditions (brightness/contrast)
    - Face size and positioning
    - Head pose estimation
    - Eye state detection
    - Motion blur detection
    - Compression artifact detection
    """

    def __init__(self, thresholds: Optional[QualityThresholds] = None):
        """
        Initialize quality assessment module.

        Args:
            thresholds: Quality thresholds configuration
        """
        self.thresholds = thresholds or QualityThresholds()

        # Initialize face/eye detectors for pose and eye state
        self.face_cascade: Optional[cv2.CascadeClassifier] = None
        self.eye_cascade: Optional[cv2.CascadeClassifier] = None
        self._initialize_detectors()

        logger.info("QualityAssessment initialized")

    def _initialize_detectors(self):
        """Initialize OpenCV cascade classifiers."""
        try:
            face_cascade_path = self._resolve_haarcascade_path(
                'haarcascade_frontalface_default.xml')
            eye_cascade_path = self._resolve_haarcascade_path(
                'haarcascade_eye.xml')

            self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
            self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
            logger.debug("Cascade classifiers initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize cascade classifiers: {e}")

    def assess_quality(self, face_image: np.ndarray) -> float:
        """
        Assess overall quality of a face image.

        Args:
            face_image: Face image as numpy array

        Returns:
            Overall quality score [0, 1]
        """
        try:
            with PerformanceTimer("quality_assessment", target_ms=50):
                result = self.assess_detailed_quality(face_image)
                return result.overall_score

        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return 0.0

    def assess_detailed_quality(self, face_image: np.ndarray) -> QualityResult:
        """
        Perform detailed quality assessment with individual metrics.

        Args:
            face_image: Face image as numpy array

        Returns:
            QualityResult with detailed analysis
        """
        try:
            metrics = {}
            passed_checks = {}
            recommendations = []

            # Convert to grayscale for some analyses
            gray = self._to_grayscale(face_image)

            # 1. Sharpness assessment
            sharpness = self._assess_sharpness(gray)
            metrics[QualityMetric.SHARPNESS.value] = sharpness
            passed_checks[QualityMetric.SHARPNESS.value] = (
                self.thresholds.min_sharpness <= sharpness <=
                self.thresholds.max_sharpness
            )
            if not passed_checks[QualityMetric.SHARPNESS.value]:
                if sharpness < self.thresholds.min_sharpness:
                    recommendations.append(
                        "Image is too blurry - ensure good focus")
                else:
                    recommendations.append(
                        "Image has excessive sharpening artifacts")

            # 2. Brightness assessment
            brightness = self._assess_brightness(gray)
            metrics[QualityMetric.BRIGHTNESS.value] = brightness
            passed_checks[QualityMetric.BRIGHTNESS.value] = (
                self.thresholds.min_brightness <= brightness <=
                self.thresholds.max_brightness
            )
            if not passed_checks[QualityMetric.BRIGHTNESS.value]:
                if brightness < self.thresholds.min_brightness:
                    recommendations.append(
                        "Image is too dark - improve lighting")
                else:
                    recommendations.append(
                        "Image is overexposed - reduce lighting")

            # 3. Contrast assessment
            contrast = self._assess_contrast(gray)
            metrics[QualityMetric.CONTRAST.value] = contrast
            passed_checks[QualityMetric.CONTRAST.value] = (
                contrast >= self.thresholds.min_contrast)
            if not passed_checks[QualityMetric.CONTRAST.value]:
                recommendations.append(
                    "Low contrast - improve lighting conditions")

            # 4. Face size assessment
            face_size = float(min(face_image.shape[:2]))
            metrics[QualityMetric.FACE_SIZE.value] = face_size
            passed_checks[QualityMetric.FACE_SIZE.value] = (
                face_size >= self.thresholds.min_face_size)
            if not passed_checks[QualityMetric.FACE_SIZE.value]:
                recommendations.append(
                    "Face too small - move closer to camera")

            # 5. Pose angle assessment
            pose_angle = self._assess_pose_angle(face_image)
            metrics[QualityMetric.POSE_ANGLE.value] = pose_angle
            passed_checks[QualityMetric.POSE_ANGLE.value] = (
                pose_angle <= self.thresholds.max_pose_angle)
            if not passed_checks[QualityMetric.POSE_ANGLE.value]:
                recommendations.append(
                    "Head pose too extreme - look more directly at camera")

            # 6. Eye openness assessment
            eye_openness = self._assess_eye_openness(face_image)
            metrics[QualityMetric.EYE_OPENNESS.value] = eye_openness
            passed_checks[QualityMetric.EYE_OPENNESS.value] = (
                eye_openness >= self.thresholds.min_eye_openness)
            if not passed_checks[QualityMetric.EYE_OPENNESS.value]:
                recommendations.append("Eyes appear closed - keep eyes open")

            # 7. Motion blur assessment
            motion_blur = self._assess_motion_blur(gray)
            metrics[QualityMetric.MOTION_BLUR.value] = motion_blur
            passed_checks[QualityMetric.MOTION_BLUR.value] = (
                motion_blur <= self.thresholds.max_motion_blur)
            if not passed_checks[QualityMetric.MOTION_BLUR.value]:
                recommendations.append("Motion blur detected - hold still")

            # 8. Compression artifacts assessment
            compression_artifacts = self._assess_compression_artifacts(
                face_image)
            metrics[QualityMetric.COMPRESSION_ARTIFACTS.value] = (
                compression_artifacts)
            passed_checks[QualityMetric.COMPRESSION_ARTIFACTS.value] = (
                compression_artifacts < 0.5)
            if not passed_checks[QualityMetric.COMPRESSION_ARTIFACTS.value]:
                recommendations.append("High compression artifacts detected")

            # Calculate overall score
            overall_score = self._calculate_overall_score(
                metrics, passed_checks)

            result = QualityResult(
                overall_score=overall_score,
                quality_metrics=metrics,
                passed_checks=passed_checks,
                recommendations=recommendations
            )

            logger.debug(f"Quality assessment: {overall_score:.3f}, "
                         f"passed: {sum(passed_checks.values())}/"
                         f"{len(passed_checks)}")

            return result

        except Exception as e:
            logger.error(f"Detailed quality assessment failed: {e}")
            return QualityResult(
                overall_score=0.0,
                quality_metrics={},
                passed_checks={},
                recommendations=["Quality assessment failed"]
            )

    def _assess_sharpness(self, gray_image: np.ndarray) -> float:
        """Assess image sharpness using Laplacian variance."""
        try:
            # Calculate Laplacian variance (higher = sharper)
            laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
            sharpness = laplacian.var()
            return float(sharpness)

        except Exception as e:
            logger.error(f"Sharpness assessment failed: {e}")
            return 0.0

    def _assess_brightness(self, gray_image: np.ndarray) -> float:
        """Assess image brightness (mean intensity)."""
        try:
            brightness = np.mean(gray_image)
            return float(brightness)

        except Exception as e:
            logger.error(f"Brightness assessment failed: {e}")
            return 0.0

    def _assess_contrast(self, gray_image: np.ndarray) -> float:
        """Assess image contrast (standard deviation of intensity)."""
        try:
            contrast = np.std(gray_image)
            return float(contrast)

        except Exception as e:
            logger.error(f"Contrast assessment failed: {e}")
            return 0.0

    def _assess_pose_angle(self, face_image: np.ndarray) -> float:
        """Estimate head pose angle using facial landmarks or
        geometric analysis."""
        try:
            # Simplified pose estimation using face symmetry
            gray = self._to_grayscale(face_image)
            h, w = gray.shape

            # Split face into left and right halves
            left_half = gray[:, :w//2]
            right_half = gray[:, w//2:]
            right_half_flipped = cv2.flip(right_half, 1)

            # Resize to same size for comparison
            min_width = min(left_half.shape[1], right_half_flipped.shape[1])
            left_resized = cv2.resize(left_half, (min_width, h))
            right_resized = cv2.resize(right_half_flipped, (min_width, h))

            # Calculate difference between halves
            diff = np.abs(left_resized.astype(np.float32) -
                          right_resized.astype(np.float32))
            asymmetry = float(np.mean(diff))

            # Convert asymmetry to approximate angle (heuristic)
            # Higher asymmetry suggests larger pose angle
            # Scale factor is heuristic
            pose_angle = min(90.0, asymmetry * 0.5)

            return float(pose_angle)

        except Exception as e:
            logger.error(f"Pose angle assessment failed: {e}")
            return 45.0  # Conservative estimate

    def _assess_eye_openness(self, face_image: np.ndarray) -> float:
        """Assess eye openness using eye detection."""
        try:
            if self.eye_cascade is None:
                return 0.5  # Neutral score if detector not available

            gray = self._to_grayscale(face_image)

            # Detect eyes
            eyes = self.eye_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10)
            )

            if len(eyes) >= 2:
                # Analyze eye regions for openness
                eye_openness_scores = []

                for (ex, ey, ew, eh) in eyes[:2]:  # Analyze first two eyes
                    eye_region = gray[ey:ey+eh, ex:ex+ew]

                    # Simple openness assessment based on intensity variation
                    # Open eyes typically have more variation due to
                    # iris/pupil contrast
                    intensity_var = float(np.var(eye_region))
                    openness = float(min(1.0, intensity_var / 500.0))
                    eye_openness_scores.append(openness)

                return float(np.mean(eye_openness_scores))
            else:
                # If eyes not detected, assume reasonable openness
                return 0.7

        except Exception as e:
            logger.error(f"Eye openness assessment failed: {e}")
            return 0.5

    def _assess_motion_blur(self, gray_image: np.ndarray) -> float:
        """Assess motion blur using gradient analysis."""
        try:
            # Calculate gradients
            grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

            # Calculate gradient magnitude
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

            # Motion blur reduces high-frequency content
            # Calculate ratio of strong edges to total gradients
            strong_edges = np.sum(gradient_magnitude > np.mean(
                gradient_magnitude) + np.std(gradient_magnitude))
            total_pixels = gradient_magnitude.size

            edge_ratio = strong_edges / total_pixels

            # Convert to blur score (lower edge ratio = more blur)
            # Scale factor is heuristic
            blur_score = 1.0 - min(1.0, edge_ratio * 10)

            return float(blur_score)

        except Exception as e:
            logger.error(f"Motion blur assessment failed: {e}")
            return 0.5

    def _assess_compression_artifacts(self, face_image: np.ndarray) -> float:
        """Assess compression artifacts using block boundary detection."""
        try:
            gray = self._to_grayscale(face_image)

            # Look for 8x8 block patterns typical in JPEG compression
            block_size = 8
            h, w = gray.shape

            # Calculate vertical and horizontal differences at block boundaries
            vertical_diffs = []
            horizontal_diffs = []

            # Check vertical block boundaries
            for i in range(block_size, h, block_size):
                if i < h - 1:
                    diff = np.mean(np.abs(gray[i, :].astype(
                        np.float32) - gray[i-1, :].astype(np.float32)))
                    vertical_diffs.append(diff)

            # Check horizontal block boundaries
            for j in range(block_size, w, block_size):
                if j < w - 1:
                    diff = np.mean(np.abs(gray[:, j].astype(
                        np.float32) - gray[:, j-1].astype(np.float32)))
                    horizontal_diffs.append(diff)

            # Calculate average boundary discontinuity
            if vertical_diffs and horizontal_diffs:
                avg_vertical = float(np.mean(vertical_diffs))
                avg_horizontal = float(np.mean(horizontal_diffs))
                avg_boundary_diff = (avg_vertical + avg_horizontal) / 2

                # Normalize to [0, 1] range
                # Scale factor is heuristic
                compression_score = min(1.0, avg_boundary_diff / 50.0)
                return float(compression_score)
            else:
                return 0.0

        except Exception as e:
            logger.error(f"Compression artifacts assessment failed: {e}")
            return 0.0

    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Ensure an image is in grayscale format."""
        if image.ndim == 2:
            return image

        if image.ndim == 3:
            channels = image.shape[2]
            if channels == 1:
                return image[:, :, 0]

            try:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            except Exception as e:
                logger.error(f"Failed to convert image to grayscale: {e}")
                raise

        raise ValueError("Unsupported image shape for grayscale conversion")

    def _resolve_haarcascade_path(self, filename: str) -> str:
        """Locate Haar cascade files in OpenCV data directory when available."""
        data_module = getattr(cv2, "data", None)
        haar_dir = getattr(data_module, "haarcascades",
                           None) if data_module else None

        if isinstance(haar_dir, str):
            candidate = Path(haar_dir) / filename
            if candidate.exists():
                return str(candidate)

        return filename

    def _calculate_overall_score(self, metrics: Dict[str, float],
                                 passed_checks: Dict[str, bool]) -> float:
        """Calculate overall quality score from individual metrics."""
        try:
            # Weight factors for different metrics
            weights = {
                QualityMetric.SHARPNESS.value: 0.25,
                QualityMetric.BRIGHTNESS.value: 0.15,
                QualityMetric.CONTRAST.value: 0.15,
                QualityMetric.FACE_SIZE.value: 0.1,
                QualityMetric.POSE_ANGLE.value: 0.15,
                QualityMetric.EYE_OPENNESS.value: 0.1,
                QualityMetric.MOTION_BLUR.value: 0.05,
                QualityMetric.COMPRESSION_ARTIFACTS.value: 0.05
            }

            # Calculate weighted score based on passed checks
            total_score = 0.0
            total_weight = 0.0

            for metric, passed in passed_checks.items():
                if metric in weights:
                    weight = weights[metric]
                    score = 1.0 if passed else 0.0

                    # For numerical metrics, also consider the actual value
                    if metric in metrics:
                        value = metrics[metric]

                        # Normalize some metrics to [0, 1] range for scoring
                        if metric == QualityMetric.SHARPNESS.value:
                            normalized = min(1.0, max(0.0, (value - 50) / 500))
                            score = score * 0.5 + normalized * 0.5
                        elif metric == QualityMetric.POSE_ANGLE.value:
                            # Lower angle = higher score
                            normalized = max(0.0, 1.0 - value / 90.0)
                            score = score * 0.5 + normalized * 0.5

                    total_score += score * weight
                    total_weight += weight

            if total_weight > 0:
                return total_score / total_weight
            else:
                return 0.0

        except Exception as e:
            logger.error(f"Overall score calculation failed: {e}")
            return 0.0

    def get_quality_recommendations(self, face_image: np.ndarray) -> List[str]:
        """Get quality improvement recommendations for an image."""
        try:
            result = self.assess_detailed_quality(face_image)
            return result.recommendations

        except Exception as e:
            logger.error(f"Quality recommendations failed: {e}")
            return ["Unable to assess quality"]


# Example usage and testing
if __name__ == "__main__":
    # Test quality assessment
    thresholds = QualityThresholds(
        min_sharpness=100.0,
        min_brightness=80.0,
        min_contrast=30.0,
        min_face_size=100
    )

    assessor = QualityAssessment(thresholds)

    print("Testing quality assessment...")

    # Create test image
    test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

    # Test quality assessment
    quality_score = assessor.assess_quality(test_image)
    print(f"Overall quality score: {quality_score:.3f}")

    # Test detailed assessment
    detailed_result = assessor.assess_detailed_quality(test_image)
    print(f"Detailed metrics: {detailed_result.quality_metrics}")
    print(f"Passed checks: {detailed_result.passed_checks}")
    print(f"Recommendations: {detailed_result.recommendations}")

    print("Quality assessment test completed!")
