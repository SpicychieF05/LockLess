"""
Feature extraction module for generating biometric templates.

This module handles deep learning-based feature extraction from face images
to create robust biometric templates for authentication.
"""

import cv2
import numpy as np
from typing import Optional, Dict, List, Sequence
try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - runtime dependency guard
    ort = None  # type: ignore[assignment]

from core.logging import get_logger, PerformanceTimer
from core.exceptions import ModelLoadError

logger = get_logger(__name__)


class FeatureExtractor:
    """
    Deep learning-based feature extractor for face recognition.

    Supports multiple architectures:
    - ArcFace (high accuracy)
    - FaceNet (balanced performance)
    - MobileFaceNet (lightweight)
    """

    def __init__(self, model_path: Optional[str] = None, architecture: str = "arcface",
                 feature_dim: int = 512, normalize_output: bool = True):
        """
        Initialize feature extractor.

        Args:
            model_path: Path to ONNX model file
            architecture: Model architecture ("arcface", "facenet", "mobilefacenet")
            feature_dim: Dimension of output features
            normalize_output: Whether to L2-normalize output features
        """
        self.architecture = architecture
        self.feature_dim = feature_dim
        self.normalize_output = normalize_output

        self.model_path = model_path or self._get_default_model_path(
            architecture)
        self.session = None
        self.input_name: Optional[str] = None
        self.input_size = (112, 112)  # Standard face recognition input size
        self.mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.std = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        self._initialize_extractor()

        logger.info(
            f"FeatureExtractor initialized with architecture: {architecture}")

    def _get_default_model_path(self, architecture: str) -> str:
        """Get default model path based on architecture."""
        model_paths = {
            "arcface": "models/face_recognition/arcface_r50.onnx",
            "facenet": "models/face_recognition/facenet_512.onnx",
            "mobilefacenet": "models/face_recognition/mobilefacenet.onnx"
        }
        return model_paths.get(architecture, model_paths["arcface"])

    def _initialize_extractor(self):
        """Initialize the feature extraction model."""
        try:
            if ort is None:
                raise ModelLoadError("ONNX Runtime not available")

            # Check if model file exists
            import os
            if not os.path.exists(self.model_path):
                logger.warning(f"Model file not found: {self.model_path}")
                # For development, create a mock extractor
                self._create_mock_extractor()
                return

            # Initialize ONNX Runtime session
            providers = ['CPUExecutionProvider']

            # Try to use GPU if available
            try:
                if ort.get_device() == 'GPU':
                    providers = ['CUDAExecutionProvider',
                                 'CPUExecutionProvider']
            except (AttributeError, RuntimeError):
                pass

            self.session = ort.InferenceSession(
                self.model_path, providers=providers)

            # Get input details
            input_details = self.session.get_inputs()[0]
            self.input_name = input_details.name
            input_shape = input_details.shape

            # Parse input shape
            if len(input_shape) == 4:  # NCHW or NHWC
                if input_shape[1] == 3:  # NCHW
                    self.input_size = (input_shape[2], input_shape[3])
                else:  # NHWC
                    self.input_size = (input_shape[1], input_shape[2])

            # Get output details
            output_details = self.session.get_outputs()[0]
            output_shape = output_details.shape
            if len(output_shape) >= 2:
                self.feature_dim = output_shape[-1]

            logger.debug(f"ONNX feature extractor loaded: {self.model_path}")
            logger.debug(
                f"Input size: {self.input_size}, Feature dim: {self.feature_dim}")

        except Exception as e:
            logger.error(f"Feature extractor initialization failed: {e}")
            logger.warning("Creating mock feature extractor for development")
            self._create_mock_extractor()

    def _create_mock_extractor(self):
        """Create a mock feature extractor for development/testing."""
        logger.info("Using mock feature extractor (for development only)")
        self.session = None  # Mark as mock
        self.input_size = (112, 112)
        self.feature_dim = 512

    def extract_features(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract features from a face image.

        Args:
            face_image: Face image as numpy array (BGR format)

        Returns:
            Feature vector as numpy array or None if extraction fails
        """
        try:
            with PerformanceTimer("feature_extraction", target_ms=100):
                if self.session is None:
                    # Use mock extractor
                    return self._extract_features_mock(face_image)
                else:
                    # Use real ONNX model
                    return self._extract_features_onnx(face_image)

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None

    def _extract_features_mock(self, face_image: np.ndarray) -> np.ndarray:
        """Mock feature extraction for development."""
        # Generate deterministic features based on image content
        # This is only for development/testing purposes

        # Resize image to standard size
        bgr_image = self._ensure_bgr(face_image)
        resized = cv2.resize(bgr_image, self.input_size)

        # Convert to grayscale and compute basic statistics
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Compute image statistics to create pseudo-features
        mean_val = float(np.mean(gray))  # type: ignore
        std_val = float(np.std(gray))  # type: ignore

        # Create feature vector based on image patches
        features = []
        patch_size = 8
        for i in range(0, resized.shape[0], patch_size):
            for j in range(0, resized.shape[1], patch_size):
                patch = gray[i:i+patch_size, j:j+patch_size]
                if patch.size > 0:
                    patch_f = patch.astype(np.float32, copy=False)
                    features.extend([
                        float(np.mean(patch_f)),
                        float(np.std(patch_f))
                    ])

        # Pad or truncate to desired dimension
        features = np.array(features[:self.feature_dim])
        if len(features) < self.feature_dim:
            # Pad with zeros
            padding = np.zeros(self.feature_dim - len(features))
            features = np.concatenate([features, padding])

        # Normalize if required
        if self.normalize_output:
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm

        return features.astype(np.float32)

    def _extract_features_onnx(self, face_image: np.ndarray) -> np.ndarray:
        """Extract features using ONNX model."""
        # Preprocess image
        input_image = self._preprocess_image(face_image)

        if self.session is None:
            raise ModelLoadError("ONNX session not initialized")

        if not self.input_name:
            raise ModelLoadError("ONNX session input name unavailable")

        # Run inference
        outputs = self.session.run(None, {self.input_name: input_image})

        # Extract features from output
        features = np.asarray(outputs[0]).astype(np.float32).flatten()

        # Normalize if required
        if self.normalize_output:
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm

        return features.astype(np.float32, copy=False)

    def _preprocess_image(self, face_image: np.ndarray) -> np.ndarray:
        """Preprocess face image for feature extraction."""
        # Resize to model input size
        bgr_image = self._ensure_bgr(face_image)
        resized = cv2.resize(bgr_image, self.input_size)

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        normalized = rgb_image.astype(np.float32) / 255.0

        # Apply mean and std normalization
        normalized = (normalized - self.mean) / self.std

        # Convert to NCHW format
        input_image = np.transpose(normalized, (2, 0, 1))
        input_image = np.expand_dims(input_image, axis=0)

        return input_image

    def compute_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """
        Compute cosine similarity between two feature vectors.

        Args:
            features1: First feature vector
            features2: Second feature vector

        Returns:
            Cosine similarity score [-1, 1]
        """
        try:
            # Ensure features are normalized
            norm1 = np.linalg.norm(features1)
            norm2 = np.linalg.norm(features2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            # Compute cosine similarity
            similarity = float(np.dot(features1, features2) / (norm1 * norm2))

            # Clamp to [-1, 1] range
            similarity = float(max(-1.0, min(1.0, similarity)))

            return similarity

        except Exception as e:
            logger.error(f"Similarity computation failed: {e}")
            return 0.0

    def compute_distance(self, features1: np.ndarray, features2: np.ndarray,
                         metric: str = "cosine") -> float:
        """
        Compute distance between two feature vectors.

        Args:
            features1: First feature vector
            features2: Second feature vector
            metric: Distance metric ("cosine", "euclidean", "manhattan")

        Returns:
            Distance value (lower = more similar)
        """
        try:
            if metric == "cosine":
                # Cosine distance = 1 - cosine similarity
                similarity = self.compute_similarity(features1, features2)
                return 1.0 - similarity

            elif metric == "euclidean":
                return float(np.linalg.norm(features1 - features2))

            elif metric == "manhattan":
                return float(np.sum(np.abs(features1 - features2)))

            else:
                raise ValueError(f"Unknown distance metric: {metric}")

        except Exception as e:
            logger.error(f"Distance computation failed: {e}")
            return float('inf')

    def batch_extract_features(self, face_images: Sequence[np.ndarray]) -> List[Optional[np.ndarray]]:
        """
        Extract features from multiple face images.

        Args:
            face_images: Iterable of face images

        Returns:
            List of feature vectors (None when extraction fails)
        """
        features_list: List[Optional[np.ndarray]] = []

        for i, face_image in enumerate(face_images):
            features = self.extract_features(face_image)
            if features is not None:
                features_list.append(features)
            else:
                logger.warning(f"Feature extraction failed for image {i}")
                features_list.append(None)

        return features_list

    def get_feature_statistics(self, features: np.ndarray) -> Dict[str, float]:
        """
        Get statistics about a feature vector.

        Args:
            features: Feature vector

        Returns:
            Dictionary with statistics
        """
        try:
            stats = {
                "dimension": int(features.size),
                "mean": float(np.mean(features)),
                "std": float(np.std(features)),
                "min": float(np.min(features)),
                "max": float(np.max(features)),
                "norm": float(np.linalg.norm(features)),
                "sparsity": float(np.sum(features == 0) / max(1, features.size))
            }
            return stats

        except Exception as e:
            logger.error(f"Feature statistics computation failed: {e}")
            return {}

    def validate_features(self, features: Optional[np.ndarray]) -> bool:
        """
        Validate feature vector quality.

        Args:
            features: Feature vector to validate

        Returns:
            True if features are valid
        """
        try:
            # Check for None or empty
            if features is None or features.size == 0:
                return False

            # Check dimension
            if features.size != self.feature_dim:
                logger.warning(
                    f"Feature dimension mismatch: {features.size} != {self.feature_dim}")
                return False

            # Check for NaN or inf values
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                logger.warning("Features contain NaN or inf values")
                return False

            # Check for zero vector
            if np.allclose(features, 0.0):
                logger.warning("Features are zero vector")
                return False

            # Check for reasonable value range
            if np.max(np.abs(features)) > 100:
                logger.warning("Features have unusually large values")
                return False

            return True

        except Exception as e:
            logger.error(f"Feature validation failed: {e}")
            return False

    def _ensure_bgr(self, image: np.ndarray) -> np.ndarray:
        """Convert input image to 3-channel BGR format if necessary."""
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        if image.ndim == 3:
            if image.shape[2] == 3:
                return image
            if image.shape[2] == 1:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        raise ValueError("Unsupported image format for feature extraction")


# Example usage and testing
if __name__ == "__main__":
    # Test feature extraction
    extractor = FeatureExtractor(architecture="arcface")

    print("Testing feature extraction...")

    # Create a test face image
    test_image = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)

    # Extract features
    features = extractor.extract_features(test_image)

    if features is not None:
        print(f"Extracted features: {features.shape}")
        print(f"Feature dimension: {len(features)}")

        # Validate features
        is_valid = extractor.validate_features(features)
        print(f"Features valid: {is_valid}")

        # Get statistics
        stats = extractor.get_feature_statistics(features)
        print(f"Feature statistics: {stats}")

        # Test similarity computation
        features2 = extractor.extract_features(test_image)
        if features2 is not None:
            similarity = extractor.compute_similarity(features, features2)
            print(f"Self-similarity: {similarity:.4f}")
        else:
            print("Failed to extract features for similarity test")

    else:
        print("Feature extraction failed")

    print("Feature extraction test completed!")
