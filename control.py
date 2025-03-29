from mavlink_wrapper import MavlinkWrapper
from camera import Camera
from pymavlink import mavutil
from red_tergat_detection import find_red_spot_center
from pid_ff_controller import PID_FF_controller
import time

if __name__ == '__main__':
    connection_string = '/dev/ttyACM0'  
    # connection_string = "udpin:localhost:14551"
    mavlink_wrapper = MavlinkWrapper(connection_string)
    mavlink_wrapper.connect()
    mavlink_wrapper.run_telemetry_parralel()
    # mavlink_wrapper.set_message_rate(mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, 1)
    cam = Camera(type="rpi", video_path=None, camera_id="/dev/video0")

    pid_x = PID_FF_controller(Kp = 1, Ki = 1,Kd = 0, Kff = 0, i_max = 1, min_cmd = 1000, max_cmd = 2000)
    pid_y = PID_FF_controller(Kp = 1, Ki = 1, Kd = 0, Kff = 0, i_max = 1, min_cmd = 1000, max_cmd = 2000)

    ret, frame = cam.get_frame()
    if not ret:
        print("Error reading frame")
    image_y_center = frame.shape[0]/2
    image_x_center = frame.shape[1]/2
    
    while True:
        ret, frame = cam.get_frame()
        if not ret:
            print("Error reading frame")
            break
        coord, mask_cleaned = find_red_spot_center(frame)
        print(coord)
        dx = image_x_center - coord[0]
        dy = image_y_center - coord[1]
        cmd_x = pid_x.get_command(setpoint = 0,current_value = dx,current_time = time.time())
        cmd_y = pid_y.get_command(setpoint = 0,current_value = dy,current_time = time.time())

        mavlink_wrapper.set_rc_channel_pwm(channel_id = 1, pwm=cmd_x)
        mavlink_wrapper.set_rc_channel_pwm(channel_id = 0, pwm=cmd_y)


