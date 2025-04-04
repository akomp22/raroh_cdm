from mavlink_wrapper import MavlinkWrapper
from camera import Camera
from pymavlink import mavutil
from target_detection import find_red_spot_center
from pidff_controller import PIDFFController
import time
import numpy as np

if __name__ == '__main__':
    connection_string = '/dev/ttyACM0'  
    # connection_string = "udpin:localhost:14551"
    source_system = 255
    coord_alpha = 0.5

    mavlink_wrapper = MavlinkWrapper(connection_string, source_system = source_system, data_list = ['AOA_SSA'])
    mavlink_wrapper.connect()
    mavlink_wrapper.run_telemetry_parralel()

    fx = 1500
    fy = 1500
    cx = 320
    cy = 240

    ###### Somehow this initiates god connection ##########
    mavlink_wrapper.connection.mav.rc_channels_override_send(
        mavlink_wrapper.connection.target_system,
        mavlink_wrapper.connection.target_component,
        1500, 1500, 1500, 1500, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0  # Fill rest with zeros
    )
    ##################################################

    # rc1_trim = mavlink_wrapper.read_parameter('RC1_TRIM')
    # rc2_trim = mavlink_wrapper.read_parameter('RC2_TRIM')
    rc1_trim = 1500
    rc2_trim = 1500
    print(f"RC1_TRIM: {rc1_trim}, RC2_TRIM: {rc2_trim}")

    cam = Camera(type="rpi", camera_id="/dev/video0", video_path=None, resolution=(640, 480))

    pid_ch1 = PIDFFController(Kp = 2, Ki = 0,Kd = 0, Kff = 0, i_max = 1, nonlinear_mode="squeared")
    pid_ch2 = PIDFFController(Kp = 2, Ki = 0, Kd = 0, Kff = 0, i_max = 1, nonlinear_mode="squeared")

    ret, frame = cam.get_frame()
    
    prev_coord = (0, 0)
    while True:
        ret, frame = cam.get_frame()
        coord, mask_cleaned = find_red_spot_center(frame)
        last_seen_time = time.time()
        if coord:
            last_coord = coord
            last_seen_time = time.time()
        elif time.time() - last_seen_time < 0.5:
            coord = last_coord
        else:
            continue
        
        coord_filtered = coord_alpha * np.array(coord) + (1 - coord_alpha) * np.array(prev_coord)
        prev_coord = coord_filtered

        angle_x_rad = np.arctan(coord_filtered[0] / fx)  # Horizontal angle
        angle_y_rad = np.arctan(coord_filtered[1] / fy)  # Vertical angle

        # aoa_ssa = mavlink_wrapper.messages["AOA_SSA"]
        # aoa = aoa_ssa.aoa
        # ssa = aoa_ssa.ssa


        cmd_x = pid_ch1.get_command(setpoint = 0, current_value = angle_x_rad, current_time = time.time())
        cmd_y = pid_ch2.get_command(setpoint = 0, current_value = angle_y_rad, current_time = time.time())

        cmd_x = rc1_trim+cmd_x
        cmd_y = rc2_trim+cmd_y
        cmd_x = max(min(cmd_x, 2000), 1000)
        cmd_y = max(min(cmd_y, 2000), 1000)
        print(f"target x {coord[0]} target y{coord[1]}; {cmd_x}; cmd y {cmd_y}")
        mavlink_wrapper.set_rc_channel_pwm([1,2], [int(cmd_x), int(cmd_y)])



