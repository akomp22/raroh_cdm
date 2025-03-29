import cv2
import numpy as np
from camera import Camera 

def find_red_spot_center(frame):
    # Convert to RGB (OpenCV loads in BGR by default)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Define red color range in RGB
    lower_red = np.array([150, 0, 0])
    upper_red = np.array([255, 100, 100])

    # Create a mask for red regions
    mask = cv2.inRange(rgb, lower_red, upper_red)

    # Morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_DILATE, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy), mask_cleaned
    return None, mask_cleaned

if __name__ == "__main__":
    import sys
    import time
    cam = Camera(type="rpi", video_path=None, camera_id="/dev/video0")

    ret, frame = cam.get_frame()
    if not ret:
        print("Error: Could not read first frame.")
        cam.release()
        sys.exit(1)

    try:
        while True:
            ret, frame = cam.get_frame()
            if not ret:
                print("Error reading frame")
                break
            
            coord, mask_cleaned = find_red_spot_center(frame)
            print(np.max(frame))
            print(coord)

            # Draw a circle at the red spot center (if found)
            # if coord:
            #     cv2.circle(frame, coord, 5, (0, 255, 0), -1)

            # # Show the original frame and mask
            # cv2.imshow("Camera Frame", frame)
            # cv2.imshow("Red Mask", mask_cleaned)

            # # Exit if 'q' is pressed
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

    except KeyboardInterrupt:
        print("\nRecording interrupted by user.")

    finally:
        cam.release()
        cv2.destroyAllWindows()
