import os
import time
import json
from datetime import datetime
import pandas as pd
import cv2

def json_to_excel(json_path, excel_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Create a DataFrame for each scalar and set index to 'step'
    dfs = []
    for scalar_name, entries in data.items():
        df = pd.DataFrame(entries)
        df = df.set_index("step")
        df = df.rename(columns={
            "timestamp": f"{scalar_name}_timestamp",
            "value": f"{scalar_name}_value"
        })
        df = df[[f"{scalar_name}_timestamp", f"{scalar_name}_value"]]
        df[f"{scalar_name}_step"] = df.index  # Add step as a column for consistency
        dfs.append(df)

    # Merge all dataframes on step index
    result = pd.concat(dfs, axis=1)

    # Move all _step columns to the correct positions (next to their corresponding value)
    ordered_cols = []
    for scalar_name in data.keys():
        ordered_cols.extend([
            f"{scalar_name}_timestamp",
            f"{scalar_name}_step",
            f"{scalar_name}_value"
        ])
    result = result[ordered_cols]

    # Reset index and export
    result.reset_index(drop=True, inplace=True)
    result.to_excel(excel_path, index=False)



from multiprocessing import Process, Queue
from datetime import datetime
import json
import time
import os
import cv2


class Logger:
    def __init__(self, base_log_dir="flight_logs"):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(base_log_dir, timestamp)
        os.makedirs(self.log_dir, exist_ok=True)

        # Scalar Logging
        self.scalar_log_path = os.path.join(self.log_dir, "scalars.json")
        self.scalar_queue = Queue()
        self.scalar_writer = Process(target=self._scalar_writer_loop, args=(self.scalar_queue, self.scalar_log_path))
        self.scalar_writer.start()

        # Video Logging
        self.video_queue = Queue()
        self.video_writer = Process(target=self._video_writer_loop, args=(self.video_queue, self.log_dir))
        self.video_writer.start()

        self.video_writers = {}  # not used here anymore
        self.video_info = {}     # just for completeness

    def add_scalar(self, name, value, step):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "step": step,
            "value": value
        }
        self.scalar_queue.put((name, log_entry))
        # print(f"[Logger] Queued: {name} @ step {step}")

    def _scalar_writer_loop(self, queue, path):
        scalars = {}
        last_flush = time.time()
        flush_interval = 1.0

        while True:
            try:
                name, entry = queue.get(timeout=0.1)

                if name == "__STOP__":
                    break

                if name not in scalars:
                    scalars[name] = []
                scalars[name].append(entry)

            except:
                pass  # queue timeout

            if time.time() - last_flush > flush_interval:
                with open(path, 'w') as f:
                    json.dump(scalars, f, indent=2)
                last_flush = time.time()

        # Final flush
        with open(path, 'w') as f:
            json.dump(scalars, f, indent=2)


    def add_frame_to_video(self, video_name, frame, fps=30):
        # Compress the frame using JPEG
        success, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if success:
            try:
                self.video_queue.put_nowait((video_name, encoded.tobytes(), fps))
            except:
                pass  # Queue full? Drop frame silently
    def _video_writer_loop(self, queue, log_dir):
        import cv2
        import numpy as np
        import os

        writers = {}
        video_fps = {}

        while True:
            try:
                video_name, encoded_bytes, fps = queue.get(timeout=0.2)

                if video_name == "__STOP__":
                    break

                # Decode the frame
                frame = cv2.imdecode(np.frombuffer(encoded_bytes, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                height, width = frame.shape[:2]

                if video_name not in writers:
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    video_path = os.path.join(log_dir, f"{video_name}.avi")
                    writers[video_name] = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
                    video_fps[video_name] = fps

                writers[video_name].write(frame)

            except Exception as e:
                continue  # Timeouts or decoding errors

        # Release all writers on shutdown
        for writer in writers.values():
            writer.release()
        print("[VideoWriter] Clean exit")


    def log_params(self, params: dict):
        params_path = os.path.join(self.log_dir, "params.json")
        with open(params_path, 'w') as f:
            json.dump(params, f, indent=2)
            
    def close(self):
        try:
            self.scalar_queue.put(("__STOP__", None))
            self.video_queue.put(("__STOP__", None, None))
        except:
            pass

        self.scalar_writer.join(timeout=5)
        self.video_writer.join(timeout=5)

        self.scalar_queue.close()
        self.video_queue.close()



if __name__ == "__main__":
    import numpy as np

    logger = Logger()

    # Simulate logging
    for step in range(100):
        logger.add_scalar("loss", 0.01 * step, step)
        if step % 10 == 0:
            logger.add_scalar("accuracy", 0.1 * step, step)

        # Dummy frame (black with a white square that moves)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.rectangle(frame, (10 + step * 5, 50), (60 + step * 5, 100), (255, 255, 255), -1)

        logger.add_frame_to_video("camera_view", frame)
        logger.add_frame_to_video("secondary_view", frame)

    logger.close()

