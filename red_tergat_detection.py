import cv2
import numpy as np
from camera import Camera

def find_red_spot_center(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]

    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            height, width = frame.shape[:2]
            return (cx - width // 2, cy - height // 2), mask_cleaned
    return None, mask_cleaned

if __name__ == "__main__":
    import sys
    import os
    import time

    cam = Camera(type="rpi", camera_id="/dev/video0",video_path=None)
    ret, frame = cam.get_frame()
    height, width = frame.shape[:2]
    print(height, width)
    time.sleep(5)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    if not os.path.exists("test_videos"):
        os.makedirs("test_videos")
    out_frame = cv2.VideoWriter("test_videos/camera_output.avi", fourcc, 30.0, (width, height))
    out_mask = cv2.VideoWriter("test_videos/mask_output.avi", fourcc, 30.0, (width, height), isColor=False)
    frame_count = 0
    try:
        while True:
            ret, frame = cam.get_frame()
            coord, mask_cleaned = find_red_spot_center(frame)
            print(f"Coordinates: {coord}")
            if coord:
                disp_coord = (coord[0] + width // 2, coord[1] + height // 2)
                cv2.circle(frame, coord, 5, (0, 255, 0), -1)
            out_frame.write(frame)
            out_mask.write(mask_cleaned)
            frame_count = frame_count + 1

            # # Optional: Show live preview
            # cv2.imshow("Camera Frame", frame)
            # cv2.imshow("Red Mask", mask_cleaned)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

    except KeyboardInterrupt:
        print("\nRecording interrupted by user.")

    finally:
        print(f"Total frames processed: {frame_count}")
        cam.release()
        out_frame.release()
        out_mask.release()
        cv2.destroyAllWindows()
