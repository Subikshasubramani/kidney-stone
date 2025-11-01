import cv2
from ultralytics import YOLO
import torch

class KidneyStoneDetectionModel:

    def __init__(self, model_path) -> None:
        # Load the YOLO model from the provided path
        self.model = YOLO(model=model_path)
        self.results = []

    def run_inference(self, image, conf_threshold=0.5, iou_threshold=0.4):
        """
        Run inference on an image with the given confidence and IoU thresholds.
        :param image: Input image for inference
        :param conf_threshold: Confidence threshold for predictions
        :param iou_threshold: IoU threshold for non-max suppression (NMS)
        :return: Number of stones and maximum stone size
        """
        # Set model confidence and IoU thresholds
        self.results = self.model(image, conf=conf_threshold, iou=iou_threshold)

        num_stones = 0
        max_stone_size = 0

        if self.results:
            # Get the number of detected stones
            num_stones = len(self.results[0].boxes)

            # If stones are detected, calculate the maximum stone size
            if num_stones > 0:
                max_stone_size = max(
                    [((box.xyxy[0][2] - box.xyxy[0][0]) ** 2 + (box.xyxy[0][3] - box.xyxy[0][1]) ** 2) ** 0.5
                     for box in self.results[0].boxes]
                )

        return num_stones, max_stone_size

    def annotate_image_with_sizes(self, image):
        """
        Annotate the image with detection boxes and return the stone sizes.
        :param image: Input image to annotate
        :return: Annotated image, severity level, and sizes of the detected stones
        """
        annotated_image = image.copy()
        num_stones = 0
        max_stone_size = 0
        stone_sizes = []

        # Define a list of colors for the stones
        colors = [
            (0, 255, 0),   # Green
            (255, 0, 0),   # Red
            (0, 0, 255),   # Blue
            (255, 255, 0), # Cyan
            (255, 0, 255), # Magenta
            (0, 255, 255)  # Yellow
        ]
        color_idx = 0

        for result in self.results:
            boxes = result.boxes
            if len(boxes) > 0:
                # Stones detected
                num_stones = len(boxes)
                for box in boxes:
                    coord = box.xyxy[0]
                    x1, y1, x2, y2 = map(int, coord)

                    # Use a different color for each box
                    color = colors[color_idx % len(colors)]
                    color_idx += 1

                    # Draw a thicker bounding box for visibility
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 3)

                    # Calculate size of the stone
                    stone_size = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                    stone_sizes.append(stone_size)
                    max_stone_size = max(max_stone_size, stone_size)

        severity = self.predict_severity(num_stones, max_stone_size)

        return annotated_image, severity, stone_sizes

    def predict_severity(self, num_stones, max_stone_size):
        """
        Predict the severity level based on the number and size of detected stones.
        :param num_stones: Number of detected stones
        :param max_stone_size: Maximum stone size in pixels
        :return: Severity level as a string
        """
        if num_stones == 0:
            return "None"
        elif max_stone_size < 50:
            return "Low"
        elif 50 <= max_stone_size < 100 or num_stones > 1:
            return "High"
        elif max_stone_size >= 100 or num_stones > 3:
            return "High"
        else:
            return "Moderate"


if __name__ == "__main__":
    model_path = "./ks_detection.pt"  # Path to the trained YOLO model

    print("Loading model...")
    model = KidneyStoneDetectionModel(model_path=model_path)

    img_path = "./sample_image.jpg"  # Path to a sample test image

    print("Reading image...")
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("Running inference...")
    num_stones, max_stone_size = model.run_inference(image=image, conf_threshold=0.5, iou_threshold=0.4)

    print("Annotating image with sizes...")
    annotated_image, severity, stone_sizes = model.annotate_image_with_sizes(image=image)

    print(f"Severity: {severity}")
    print(f"Stone sizes: {stone_sizes}")

    # Show the annotated image
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    cv2.imshow("Results", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
