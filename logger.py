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



class Logger:
    def __init__(self, base_log_dir="flight_logs", flush_interval=100):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(base_log_dir, timestamp)
        os.makedirs(self.log_dir, exist_ok=True)

        self.scalar_log_path = os.path.join(self.log_dir, "scalars.json")
        self.scalars = {}
        self._step_counter = 0
        self.flush_interval = flush_interval  # flush every N steps

        self.video_writers = {}
        self.video_info = {}

    def add_scalar(self, name, value, step):
        if name not in self.scalars:
            self.scalars[name] = []
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "step": step,
            "value": value
        }
        self.scalars[name].append(log_entry)

        self._step_counter += 1
        if self._step_counter >= self.flush_interval:
            self._write_scalars()
            self._step_counter = 0

    def _write_scalars(self):
        with open(self.scalar_log_path, 'w') as f:
            json.dump(self.scalars, f, indent=2)

    def close(self):
        self._write_scalars()
        for writer in self.video_writers.values():
            writer.release()

    def add_frame_to_video(self, video_name, frame, fps=30):
        if video_name not in self.video_writers:
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_path = os.path.join(self.log_dir, f"{video_name}.avi")
            writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            self.video_writers[video_name] = writer
            self.video_info[video_name] = {"fps": fps, "size": (width, height)}
        self.video_writers[video_name].write(frame)

    def log_params(self, params: dict):
        params_path = os.path.join(self.log_dir, "params.json")
        with open(params_path, 'w') as f:
            json.dump(params, f, indent=2)

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

