import cv2
import numpy as np
from camera import Camera

def find_red_spot_center(frame, cx, cy):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 120, 90])
    upper_red1 = np.array([5, 255, 255])
    lower_red2 = np.array([175, 150, 150])
    upper_red2 = np.array([180, 255, 255])


    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # kernel = np.ones((3, 3), np.uint8)
    # mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)
    mask_cleaned = mask

    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]

    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] != 0:
            x = int(M["m10"] / M["m00"])
            y = int(M["m01"] / M["m00"])
            # height, width = frame.shape[:2]
            return (x - cx, y - cy), mask_cleaned
    return None, mask_cleaned

if __name__ == "__main__":
    import sys
    import os
    import time
    from datetime import datetime

    cam = Camera(type="rpi", camera_id="/dev/video0", video_path=None, resolution=(640, 480))
    ret, frame = cam.get_frame()
    height, width = frame.shape[:2]
    cx = width // 2
    cy = height // 2
    print(height, width)
    time.sleep(2)

    # Create timestamped test folder
    test_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    test_dir = f"test_videos/{test_time}"
    os.makedirs(test_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_frame = cv2.VideoWriter(f"{test_dir}/camera_output.avi", fourcc, 30.0, (width, height))
    out_mask = cv2.VideoWriter(f"{test_dir}/mask_output.avi", fourcc, 30.0, (width, height), isColor=False)

    frame_count = 0
    start_time = time.time()
    try:
        while True:
            ret, frame = cam.get_frame()
            coord, mask_cleaned = find_red_spot_center(frame, cx, cy)

            frame_time = time.time()
            fps = 1.0 / (frame_time - start_time)
            start_time = frame_time
            timestamp_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]

            if frame_count % 10 == 0:
                print(f"Coordinates: {coord}, FPS: {fps:.2f}")

            # Draw coordinates if found
            if coord:
                disp_coord = (coord[0] + width // 2, coord[1] + height // 2)
                cv2.circle(frame, disp_coord, 5, (0, 255, 0), -1)
                cv2.circle(mask_cleaned, disp_coord, 5, (255), -1)

            # Draw frame number and timestamp
            label = f"Frame: {frame_count}, Time: {timestamp_str}"
            cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(mask_cleaned, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255), 2)

            out_frame.write(frame)
            out_mask.write(mask_cleaned)
            frame_count += 1

    except KeyboardInterrupt:
        print("\nRecording interrupted by user.")

    finally:
        print("Recording stopped.")
        cam.release()
        out_frame.release()
        out_mask.release()
        cv2.destroyAllWindows()
