from mavlink_wrapper import MavlinkWrapper
from camera import Camera
from pymavlink import mavutil
from target_detection import find_red_spot_center
from pidff_controller import PIDFFController
import time
import numpy as np
from logger import Logger
import cv2

if __name__ == '__main__':
    COORD_ALPHA = 0.99
    NAV_GAIN = 5000
    KP_CH1 = 8000
    KP_CH2 = 8000

    REVERSED_CH1 = True
    REVERSED_CH2 = False
    MAC_CH1 = 1800
    MIN_CH1 = 1200 
    MAC_CH2 = 1800
    MIN_CH1 = 1200

    SAVE_DATA = True


    cam = Camera(type="rpi", camera_id="/dev/video0", video_path=None, resolution=(640, 480))
    time.sleep(4)
    camera_matrix, dist_coeffs = Camera.read_params(folder = "params_rpi_0")
    # cam.init_undiostort(camera_matrix, dist_coeffs)

    # optimalCameraMatrix = cam.optimalCameraMatrix
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    print(f"fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}")

    pid_ch1 = PIDFFController(Kp = KP_CH1, Ki = 0,Kd = 0, Kff = 0, i_max = 1, nonlinear_mode='squared')
    pid_ch2 = PIDFFController(Kp = KP_CH2, Ki = 0, Kd = 0, Kff = 0, i_max = 1, nonlinear_mode='squared')

    logger = Logger(base_log_dir="flight_logs")
    param_dict = {
        "Kp_ch1": KP_CH1,
        "Kp_ch2": KP_CH2,
        "coord_alpha": COORD_ALPHA,
        "nav_gain": NAV_GAIN,
        "reversed_ch1": REVERSED_CH1,
        "reversed_ch2": REVERSED_CH2,
        "max_ch1": MAC_CH1,
        "min_ch1": MIN_CH1,
        "max_ch2": MAC_CH2,
        "min_ch2": MIN_CH1
    }
    logger.log_params(param_dict)

    connection_string = '/dev/ttyACM0'  
    # connection_string = "udpin:localhost:14551"
    source_system = 255
    mavlink_wrapper = MavlinkWrapper(connection_string, source_system = source_system, data_list = ['AOA_SSA'])
    mavlink_wrapper.connect()
    mavlink_wrapper.run_telemetry_parralel()


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
    prev_coord = (0, 0)
    last_coord = (0, 0)
    angle_ch1_rad_prev = 0.0
    angle_ch2_rad_prev = 0.0
    while True:
        ret, frame = cam.get_frame()
        coord, mask_cleaned = find_red_spot_center(frame, cx, cy)
        t1 = time.time()

        # droput protection
        last_seen_time = time.time()
        if coord:
            last_coord = coord
            last_seen_time = time.time()
        elif time.time() - last_seen_time < 0.5:
            coord = last_coord
        else:
            continue
        
        # Filter coordinates
        coord_filtered = COORD_ALPHA * np.array(coord) + (1 - COORD_ALPHA) * np.array(prev_coord)
        prev_coord = coord_filtered

        # calculate angle to target 
        angle_ch1_rad = np.arctan(coord_filtered[0] / fx)  # Horizontal angle
        angle_ch2_rad = np.arctan(coord_filtered[1] / fy)  # Vertical angle

        # visual guidance system
        aoa_ssa = mavlink_wrapper.messages["AOA_SSA"]
        aoa = aoa_ssa.AOA
        ssa = aoa_ssa.SSA
        cmd_ch1 = pid_ch1.get_command(setpoint = ssa, current_value = angle_ch1_rad, current_time = time.time())
        cmd_ch2 = pid_ch2.get_command(setpoint = aoa, current_value = angle_ch2_rad, current_time = time.time())

        # PN system
        dt = time.time() - t1
        d_error_ch1 = (angle_ch1_rad - angle_ch1_rad_prev)  / dt if dt > 0 else 0.0
        d_error_ch2 = (angle_ch2_rad - angle_ch2_rad_prev)  / dt if dt > 0 else 0.0

        pn_term_ch1 = -NAV_GAIN * d_error_ch1
        pn_term_ch2 = -NAV_GAIN * d_error_ch2
        cmd_ch1 += pn_term_ch1
        cmd_ch2 += pn_term_ch2
        angle_ch1_rad_prev = angle_ch1_rad
        angle_ch2_rad_prev = angle_ch2_rad

        # execute command
        if REVERSED_CH1:
            cmd_ch1 = -cmd_ch1
        if REVERSED_CH2:
            cmd_ch2 = -cmd_ch2
        cmd_ch1 = int(rc1_trim+cmd_ch1)
        cmd_ch2 = int(rc2_trim+cmd_ch2)
        cmd_ch1 = max(min(cmd_ch1, MAC_CH1), MIN_CH1)
        cmd_ch2 = max(min(cmd_ch2, MAC_CH2), MIN_CH1)
        print(f"angle_x_rad {angle_ch1_rad:.3f} angle_y_rad {angle_ch2_rad:.3f}; cmd x {cmd_ch1}; cmd y {cmd_ch2}")
        mavlink_wrapper.set_rc_channel_pwm([1,2], [cmd_ch1, cmd_ch2])

        if SAVE_DATA:
            logger.add_scalar("cmd_ch1", cmd_ch1, time.time())
            logger.add_scalar("cmd_ch2", cmd_ch2, time.time())
            logger.add_scalar("angle_ch1_rad", angle_ch1_rad, time.time())
            logger.add_scalar("angle_ch2_rad", angle_ch2_rad, time.time())
            logger.add_scalar("coord_x", coord[0], time.time())
            logger.add_scalar("coord_y", coord[1], time.time())
            logger.add_scalar("coord_x_filtered", coord_filtered[0], time.time())
            logger.add_scalar("coord_y_filtered", coord_filtered[1], time.time())
            logger.add_scalar("aoa", aoa, time.time())
            logger.add_scalar("ssa", ssa, time.time())
            logger.add_scalar("pn_term_ch1", pn_term_ch1, time.time())
            logger.add_scalar("pn_term_ch2", pn_term_ch2, time.time())
            logger.add_scalar("d_error_ch1", d_error_ch1, time.time())
            logger.add_scalar("d_error_ch2", d_error_ch2, time.time())
            logger.add_scalar("dt", dt, time.time())

            # add coord to the frame
            if coord:
                disp_coord = (int(coord[0] + cx), int(coord[1] + cy))
                cv2.circle(frame, disp_coord, 5, (0, 255, 0), -1)
            logger.add_frame_to_video("frame", frame, fps=30)
        




