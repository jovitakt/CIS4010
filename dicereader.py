import cv2
import numpy as np

def detect_dice_dots(image_path):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=50, param2=30, minRadius=5, maxRadius=30)

    # If circles are detected, count circles
    total_dots = 0
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        total_dots = len(circles)
        
        # Draw circles on the original image for visualization
        for (x, y, r) in circles:
            cv2.circle(image, (x, y), r, (0, 255, 0), 2)
            cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        
        # Save the image with detected circles
        output_path = image_path.replace(".png", "_output.png")
        cv2.imwrite(output_path, image)
    
    return total_dots

# Paths to the test images
image_paths = ["/mnt/data/DiceTest_1.png", "/mnt/data/DiceTest_2.png","/mnt/data/DiceTest_3.png"]
# Process each image and collect results
results = {}
for path in image_paths:
    results[path] = detect_dice_dots(path)

results
