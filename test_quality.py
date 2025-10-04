#!/usr/bin/env python3
"""Test quality assessment with relaxed thresholds."""

import cv2
from src.biometric.face_detection import FaceDetector
from src.biometric.quality_assessment import QualityAssessment


def main():
    print("Testing quality assessment...")

    # Initialize
    detector = FaceDetector()
    quality = QualityAssessment()

    # Override quality thresholds for testing
    quality.thresholds.min_sharpness = 50.0
    quality.thresholds.min_brightness = 50.0
    quality.thresholds.min_contrast = 20.0
    quality.thresholds.min_face_size = 50

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Camera not available")
        return

    print("📷 Camera opened, looking for faces...")
    print("Press 'q' to quit, 's' to save current analysis")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Detect faces
        faces = detector.detect_faces(frame)

        # Display frame with detections
        display_frame = frame.copy()

        if faces:
            for face in faces:
                x, y, w, h = face
                cv2.rectangle(display_frame, (x, y),
                              (x+w, y+h), (0, 255, 0), 2)

                # Extract face
                face_image = frame[y:y+h, x:x+w]

                # Assess quality
                assessment = quality.assess_detailed_quality(face_image)

                # Display quality info
                text = f"Q: {assessment.overall_score:.3f}"
                cv2.putText(display_frame, text, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                sharpness = assessment.quality_metrics.get('sharpness', 0)
                brightness = assessment.quality_metrics.get('brightness', 0)
                contrast = assessment.quality_metrics.get('contrast', 0)
                face_size = assessment.quality_metrics.get('face_size', 0)

                print(f"\rQuality: {assessment.overall_score:.3f} "
                      f"| Sharp: {sharpness:.1f} "
                      f"| Bright: {brightness:.1f} "
                      f"| Contrast: {contrast:.1f} "
                      f"| Size: {face_size:.1f}", end="")

        cv2.imshow('Face Detection Quality Test', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and faces:
            # Save detailed analysis
            face = faces[0]
            x, y, w, h = face
            face_image = frame[y:y+h, x:x+w]
            assessment = quality.assess_detailed_quality(face_image)

            print("\n\n📊 Detailed Quality Analysis:")
            print(f"Overall Quality: {assessment.overall_score:.3f}")

            sharpness = assessment.quality_metrics.get('sharpness', 0)
            brightness = assessment.quality_metrics.get('brightness', 0)
            contrast = assessment.quality_metrics.get('contrast', 0)
            face_size = assessment.quality_metrics.get('face_size', 0)

            print(
                f"Sharpness: {sharpness:.1f} "
                f"(threshold: {quality.thresholds.min_sharpness})")
            print(
                f"Brightness: {brightness:.1f} "
                f"(threshold: {quality.thresholds.min_brightness})")
            print(
                f"Contrast: {contrast:.1f} "
                f"(threshold: {quality.thresholds.min_contrast})")
            print(
                f"Face Size: {face_size:.1f} "
                f"(threshold: {quality.thresholds.min_face_size})")

            passed_count = sum(assessment.passed_checks.values())
            total_count = len(assessment.passed_checks)
            print(
                f"Passed: {passed_count}/{total_count} criteria")

            # Save face image
            cv2.imwrite('test_face.jpg', face_image)
            print("Face image saved as test_face.jpg")

    cap.release()
    cv2.destroyAllWindows()
    print("\n\n✅ Test completed")


if __name__ == "__main__":
    main()
