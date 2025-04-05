import cv2
import numpy as np
import threading
from queue import Queue
import time
import pickle

class Camera():
    def __init__(self, type, camera_id="/dev/video0", video_path=None, resolution=(640, 480)): 
        """ 'ls /dev/video*' to find the camera id """
        assert type in ['jetson', 'rpi', 'record', 'windows'], "type must be 'jetson', 'rpi', 'record', or 'windows'"
        self.type = type
        self.video_path = video_path
        self.camera_id = camera_id
        self.resolution = resolution

        if self.type == 'record':
            self.cap = cv2.VideoCapture(self.video_path)
            # Read actual resolution from video file
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.resolution = (actual_width, actual_height)

        elif self.type in ['jetson', 'windows']:
            self.cap = cv2.VideoCapture(camera_id)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

            # Validate resolution
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if (actual_width, actual_height) != self.resolution:
                print(f"[WARNING] Requested resolution {self.resolution} not supported. Using {actual_width}x{actual_height} instead.")
                self.resolution = (actual_width, actual_height)

        elif self.type == 'rpi':
            from picamera2 import Picamera2
            self.picam2 = Picamera2()
            try:
                config = self.picam2.create_video_configuration(
                    main={"size": (320, 240), "format": "RGB888"},
                    raw={"size": (2304, 1296)},
                    controls={"FrameRate": 120}
                )
                self.picam2.configure(config)
            except Exception as e:
                print(f"[ERROR] Failed to set resolution {self.resolution} for PiCamera2: {e}")
                print("[INFO] Falling back to default resolution (640x480).")
                self.resolution = (640, 480)
                config = self.picam2.create_video_configuration(
                    main={"size": self.resolution, "format": "YUV420"},
                    raw={"size": (2304, 1296)},
                    controls={"FrameRate": 120}
                )
                self.picam2.configure(config)
            print(self.picam2.camera_controls)
            self.picam2.start(show_preview=False)
            time.sleep(2)

    def release(self):
        if self.type == 'jetson' or self.type == 'record' or self.type == 'windows':
            self.cap.release()
        elif self.type == 'rpi':
            self.picam2.stop()

    def reset(self):
        self.__init__(self.type, self.camera_id, self.video_path)
    
    def get_frame(self):
        # grab to ignore camera buffer and get most recent frame
        if self.type == 'jetson' or self.type == 'record' or self.type == 'windows':
            self.cap.grab()
            ret, frame = self.cap.read()
            if not ret:
                print("Error reading frame")
            return ret,frame
        elif self.type == 'rpi':
            frame = self.picam2.capture_array('main')
            # frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # frame = frame[:480,:]
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
    
    @staticmethod
    def read_params(folder):
        with open(f'{folder}/cameraMatrix.pkl', 'rb') as f:
            cameraMatrix = pickle.load(f)
        with open(f'{folder}/dist.pkl', 'rb') as f:
            dist = pickle.load(f)
        return cameraMatrix, dist
    
def record_video(cam, output_path, video_name, fps=30):
    try:
        os.makedirs(output_path, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI format
        out = cv2.VideoWriter(output_path + video_name, fourcc, fps, cam.resolution)
        n = 0
        while True:
            n += 1
            ret, frame = cam.get_frame()
            if not ret:
                break
            # add timestamp and n to the frame
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            cv2.putText(frame, f"{timestamp}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"{n}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            out.write(frame)
        out.release()
    except Exception as e:
        print(f"Error recording video: {e}")
        out.release()


if __name__ == "__main__":
    import sys
    import os
    # input_video = "data_inputs/1.mp4"
    current_time = time.strftime("%Y%m%d_%H%M%S")
    output_path = f"flight_logs/video/{current_time}/"
    os.makedirs(output_path, exist_ok=True)


    cam = Camera(type="rpi", camera_id="/dev/video0", video_path=None, resolution=(640, 480))
    print('Recording video...')
    record_video(cam, output_path, "test.avi", fps=30)
