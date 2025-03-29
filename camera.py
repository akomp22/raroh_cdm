import cv2
import numpy as np
import threading
from queue import Queue

class Camera():
    def __init__(self, type, video_path, camera_id = "/dev/video0"): 
        """" 'ls /dev/video*' to find the camera id"""
        assert type in ['jetson', 'rpi', 'record'], "type must be 'jetson' or 'rpi' or 'record'"
        self.type = type
        self.video_path = video_path
        self.camera_id = camera_id
        if self.type == 'record':
            self.cap = cv2.VideoCapture(self.video_path)
        elif self.type == 'jetson':
            self.cap = cv2.VideoCapture(camera_id)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        elif self.type == 'rpi':
            from picamera2 import Picamera2
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
            self.picam2.configure(config)
            config["controls"] = {"FrameRate": 50}  # Optional: Adjust frame rate
            self.picam2.start()

    def release(self):
        if self.type == 'jetson' or self.type == 'record':
            self.cap.release()
        elif self.type == 'rpi':
            self.picam2.stop()

    def reset(self):
        self.__init__(self.type, self.video_path, self.camera_id)
        
    
    def get_frame(self):
        # grab to ignore camera buffer and get most recent frame
        if self.type == 'jetson' or self.type == 'record':
            self.cap.grab()
            ret, frame = self.cap.read()
            if not ret:
                print("Error reading frame")
            return ret,frame
        elif self.type == 'rpi':
            frame = self.picam2.capture_array()
            frame = frame[:480,:]
            return True, frame
    
    def init_undiostort(self, cameraMatrix, dist):
        # Get the camera parameters
        self.cameraMatrix = cameraMatrix
        self.dist = dist
        ret,frame = self.get_frame()
        self.h,  self.w = frame.shape[:2]
        self.optimalCameraMatrix, self.roi1 = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (self.w,self.h), 1, (self.w,self.h))
        self.x_roi, self.y_roi, self.w_roi, self.h_roi = self.roi1
        # reset frame generator because first frame was used to get image size
        self.reset()

    def undistort(self, frame):
        # undistort segmentation mask
        frame_undist = cv2.undistort(frame, self.cameraMatrix, self.dist, None, self.optimalCameraMatrix)
        frame_undist_roi = np.zeros_like(frame_undist)
        frame_undist_roi[self.y_roi:self.y_roi+self.h_roi, self.x_roi:self.x_roi+self.w_roi] = frame_undist[self.y_roi:self.y_roi+self.h_roi, 
                                                                                                                self.x_roi:self.x_roi+self.w_roi]
        return frame_undist_roi
    


if __name__ == "__main__":
    import sys
    input_video = "data_inputs/1.mp4"
    output_video = "data_outputs/1.avi"

    cam = Camera(type="record",video_path=input_video, camera_id="1")

    ret, frame = cam.get_frame()
    if not ret:
        print("Error: Could not read first frame.")
        cam.release()
        sys.exit(1)

    # Get frame width & height dynamically
    frame_width = int(cam.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cam.cap.get(cv2.CAP_PROP_FPS) or 25)  # Default to 25 FPS if unknown

    # Define VideoWriter codec and output file
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # XVID for AVI format
    out = cv2.VideoWriter(output_video, fourcc, fps, (320, 240))

    print(f"Recording started: {output_video} ({frame_width}x{frame_height} @ {fps} FPS)")


    import time

    try:
        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cam.get_frame()
            
            if not ret:
                print("Error reading frame")
                break
            
            out.write(frame)  # Save frame to video file
            frame_count += 1

            # Print FPS every second
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                print(f"FPS: {fps:.2f}")
                frame_count = 0
                start_time = time.time()

    except KeyboardInterrupt:
        print("\nRecording interrupted by user.")

    finally:
        print("Recording stopped.")
        cam.release()
        out.release()
