from pymavlink import mavutil
import pymavlink.dialects.v20.all as dialects
import time
from multiprocessing import Process, Manager, Event  
import numpy as np
# from utils import *
import copy

def to_quaternion(roll=0.0, pitch=0.0, yaw=0.0):
    """
    Convert degrees to quaternions
    """
    t0 = np.cos(np.radians(yaw * 0.5))
    t1 = np.sin(np.radians(yaw * 0.5))
    t2 = np.cos(np.radians(roll * 0.5))
    t3 = np.sin(np.radians(roll * 0.5))
    t4 = np.cos(np.radians(pitch * 0.5))
    t5 = np.sin(np.radians(pitch * 0.5))

    w = t0 * t2 * t4 + t1 * t3 * t5
    x = t0 * t3 * t4 - t1 * t2 * t5
    y = t0 * t2 * t5 + t1 * t3 * t4
    z = t1 * t2 * t4 - t0 * t3 * t5
    return [w, x, y, z]

class MavlinkWrapper:
    def __init__(self, connection_string, source_system=1):
        self.connection_string = connection_string
        self.source_system = source_system
        self.message_hook_function = None

        self.manager = Manager()
        self.messages = self.manager.dict({
            'GLOBAL_POSITION_INT': None,
            'ATTITUDE': None,
            'VFR_HUD': None,
            'HEARTBEAT': None,
            'AOA_SSA': None,
            'LOCAL_POSITION_NED': None,
            'RC_CHANNELS': None,
            'COMMAND_ACK': None,
            'PARAM_VALUE': None,
            'HOME_POSITION': None,
            'MISSION_ITEM_INT': None,
            'MISSION_COUNT': None,
            'MISSION_ACK' : None,
            'MISSION_CURRENT' : None,
            'TERRAIN_REPORT' : None
        })

        self.parralel_telemetry = False

        # Events for waiting on message updates
        self.message_events = {msg_type: Event() for msg_type in self.messages.keys()}

    def connect(self):
        self.connection =  mavutil.mavlink_connection(self.connection_string, source_system=self.source_system)
        print("Connecting to vehicle on: %s" % self.connection_string)
        try:
            self.connection.wait_heartbeat(timeout=10)
            print("Heartbeat from system (system %u component %u)" % (self.connection.target_system, self.connection.target_component))
            return True
        except Exception as e:
            print(f"Error waiting for heartbeat: {e}")
            return False


    def close(self):
        # close mavlink connection
        self.connection.close()
        
    def set_message_hook(self, hook_fn):
        """Allows an external function to be set as a hook."""
        self.message_hook_function = hook_fn

    def run_telemetry(self):
        try:
            while True:
                msg = self.connection.recv_msg()
                if msg is not None:
                    msg_type = msg.get_type()
                    if msg_type in self.messages:
                        self.messages[msg_type] = msg
                        # Set the event to signal that the message has been updated
                        self.message_events[msg_type].set()
                        # If a hook function is defined, call it and pass the message
                        if self.message_hook_function:
                            self.message_hook_function(msg)

        except KeyboardInterrupt:
            print(f"Error in telemetry:")

    def initiate_wait_for_update(self, message_type):
        """Initiate a wait for a specific message type to be updated."""
        if message_type in self.message_events:
            event = self.message_events[message_type]
            # Clear the event before waiting
            event.clear()
            return True
        return False
    
    def wait_for_update(self, message_type, initiated = True, timeout=5):
        """Wait for a specific message type to be updated within the specified timeout."""
        if not initiated:
            initiated = self.initiate_wait_for_update(message_type)
            if not initiated:
                return False
        if message_type in self.message_events:
            event = self.message_events[message_type]
            # Wait for the event to be set, indicating that the message has been updated
            return event.wait(timeout)
        return False

    def run_telemetry_parralel(self):
        """Starts a separate process to run telemetry."""
        self.telemetry_parralel_process = Process(target=self.run_telemetry, args=())
        self.telemetry_parralel_process.start()
        self.parralel_telemetry = True

        # return process to manage it later (check if its alive or terminate it ,process.start(), process.terminate(),  process.join())
        return self.telemetry_parralel_process
    
    def stop_telemetry_process(self):
        """Stops the telemetry process."""
        self.telemetry_parralel_process.terminate()
        #self.telemetry_parralel_process.join()
        self.parralel_telemetry = False

    def join_telemetry_process(self):
        """Joins the telemetry process."""
        self.telemetry_parralel_process.join()
        self.parralel_telemetry = False

    def remember_instant_messages(self):
        """Remember the current values of the messages"""
        self.messages_remembered = copy.deepcopy(self.messages)


    def read_rc_channel(self,channel_number,timeout = 5):
        t1 = time.time()
        while True:
            if time.time() - t1 > timeout:
                print('Timeout recieveing rc channel')
                return None
            if self.parralel_telemetry:
                ret = self.wait_for_update('RC_CHANNELS', timeout=timeout/5)
                if ret:
                    message = self.messages['RC_CHANNELS']
                else:
                    continue
                if message is None:
                    continue
            else:
                message = self.connection.recv_match(type='RC_CHANNELS', blocking=True,timeout = timeout/5)
            if message is not None:
                # Assuming we want the value of the first RC channel as an example
                # RC channel values are typically in the range 1000 (minimum) to 2000 (maximum)
                # with ~1500 being the midpoint for many systems
                if channel_number == 1:
                    rc_channel_value = message.chan1_raw
                elif channel_number == 2:
                    rc_channel_value = message.chan2_raw
                elif channel_number == 3:
                    rc_channel_value = message.chan3_raw
                elif channel_number == 4:
                    rc_channel_value = message.chan4_raw
                elif channel_number == 5:
                    rc_channel_value = message.chan5_raw
                elif channel_number == 6:
                    rc_channel_value = message.chan6_raw
                elif channel_number == 7:
                    rc_channel_value = message.chan7_raw
                elif channel_number == 8:
                    rc_channel_value = message.chan8_raw
                if rc_channel_value == None:
                    print('Error reading rc channel value')
                if rc_channel_value is not None:
                    return rc_channel_value
                
    def set_message_rate(self,message_id, rate,check = False, check_timeout = 2):
        """ Set message rate """
        # https://mavlink.io/en/messages/common.html#MAV_CMD_SET_MESSAGE_INTERVAL
        # https://mavlink.io/en/messages/common.html#MAVLink
        # https://mavlink.io/en/messages/common.html#MAV_DATA_STREAM
        # https://mavlink.io/en/messages/common.html#MAV_CMD_SET_MESSAGE_INTERVAL
        # mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE,    mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED


        message = self.connection.mav.command_long_encode(
                                                    self.connection.target_system,          # Target system (1 for the autopilot)
                                                    self.connection.target_component,       # Target component (0 for all components)
                                                    mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                                                    0,
                                                    message_id,
                                                    rate,0,0,0,0,0)
        if check and self.parralel_telemetry:
            self.initiate_wait_for_update('COMMAND_ACK')

        self.connection.mav.send(message)
        
        if check == True:
            message_name = "set_message_rate (message id:" + str(message_id) + ")"
            r = self.check_command_ack(message_name,mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, timeout = check_timeout)
            return r
        else:
            return True
        
    def check_command_ack(self,message_name,message_id, timeout = 10):
        """ Check if the command was acknowledged"""
        t1 = time.time()
        while True:
            if time.time() - t1 > timeout:
                print(f"Failed to receive COMMAND_ACK for command {message_name} within the timeout period {timeout} s.")
                return False
            if self.parralel_telemetry:
                ret = self.wait_for_update('COMMAND_ACK', timeout=timeout/5)
                if ret:
                    msg = self.messages['COMMAND_ACK']
                else:
                    continue
                if msg is None:
                    continue
            else :
                msg = self.connection.recv_match(type = "COMMAND_ACK",blocking=True, timeout = timeout/5)
                if msg is None:
                    # print(f"Failed to receive COMMAND_ACK for command {message_name} within the timeout period {timeout/5} s.")
                    continue
                    # return False
            msg = msg.to_dict()
            if msg['command'] != message_id:
                continue
            if msg["result"] == 0:
                print(f"{message_name} successful")
                return True
            else:
                print(f"{message_name} failed")
                print(mavutil.mavlink.enums['MAV_RESULT'][msg['result']].description)
                return False

    def arm_disarm(self,ARM = True):
        if ARM:
            if self.parralel_telemetry:
                self.initiate_wait_for_update('COMMAND_ACK')

            self.connection.mav.command_long_send(self.connection.target_system,
                                        self.connection.target_component,
                                        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,     # command
                                        0,                                                # confirmation
                                        1,                                                # param 1, 0: disarm, 1: arm
                                        0,                                                # param 2, 0: arm-disarm unless prevented by safety checks (i.e. when landed), 21196: force arming/disarming (e.g. allow arming to override preflight checks and disarming in flight).
                                        0, 0, 0,0,0)                                      # param 3 ~ 7 not used
            # msg = self.connection.recv_match(type = "COMMAND_ACK",blocking=True)
            # msg = msg.to_dict()

            r = self.check_command_ack(message_name="ARM",message_id=mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, timeout = 10)
            return r
        else:
            if self.parralel_telemetry:
                self.initiate_wait_for_update('COMMAND_ACK')
            self.connection.mav.command_long_send(self.connection.target_system,
                                        self.connection.target_component,
                                        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,     # command
                                        0,                                                # confirmation
                                        0,                                                # param 1, 0: disarm, 1: arm
                                        0,                                                # param 2, 0: arm-disarm unless prevented by safety checks (i.e. when landed), 21196: force arming/disarming (e.g. allow arming to override preflight checks and disarming in flight).
                                        0, 0, 0,0,0)                                      # param 3 ~ 7 not used
            r = self.check_command_ack(message_name="DISARM",message_id=mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, timeout = 10)
            return  r

    def request_message(self,message_id,check = False):
        """ Request message. Needs to be chacked in wrapper """
        request_message_command = dialects.MAVLink_command_long_message(target_system=self.connection.target_system,
                                                                    target_component=self.connection.target_component,
                                                                    command=dialects.MAV_CMD_REQUEST_MESSAGE,
                                                                    confirmation=0,
                                                                    param1=message_id,
                                                                    param2=0,                      # not used
                                                                    param3=0,                      # not used
                                                                    param4=0,                      # not used
                                                                    param5=0,                      # not used
                                                                    param6=0,                      # not used
                                                                    param7=0                       # Target address for requested message (if message has target address fields). 0: Flight-stack default, 1: address of requestor, 2: broadcast. 
                                                                    )               

        # send command to the vehicle
        if check and self.parralel_telemetry:
            self.initiate_wait_for_update('COMMAND_ACK')

        self.connection.mav.send(request_message_command)
        if check == True:
            message_name = "request_message (messgae id:" + str(message_id) + ")"
            r = self.check_command_ack(message_name,dialects.MAV_CMD_REQUEST_MESSAGE, timeout = 2)
            return r
        else:
            return True
        
    def set_rc_channel_pwm(self,channel_id, pwm=1500):
        """ Set RC channel pwm value
        Args:
            channel_id (TYPE): Channel ID
            pwm (int, optional): Channel pwm value 1100-1900
        """
        if channel_id < 1 or channel_id > 18:
            print("Channel does not exist.")
            return False

        # Mavlink 2 supports up to 18 channels:
        # https://mavlink.io/en/messages/common.html#RC_CHANNELS_OVERRIDE
        rc_channel_values = [0 for _ in range(16)]
        rc_channel_values[channel_id - 1] = pwm
        self.connection.mav.rc_channels_override_send(
                self.connection.target_system,                # target_system
                self.connection.target_component,             # target_component
                *rc_channel_values)                  # RC channel list, in microseconds.
        
    def set_mode(self,mode,timeout = 10):
        if mode not in self.connection.mode_mapping():
            print('Unknown mode : {}'.format(mode))
            print('Try:', list(self.connection.mode_mapping().keys()))
            return False
        mode_id = self.connection.mode_mapping()[mode]

        if self.parralel_telemetry:
            self.initiate_wait_for_update('HEARTBEAT')

        self.connection.mav.set_mode_send(
                                self.connection.target_system,
                                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                                mode_id)
        t1 = time.time()
        while True:
            if time.time() - t1 > timeout:
                print(f"Failed to receive HEARTBEAT within the timeout period {timeout} s.")
                return False
            if self.parralel_telemetry:
                ret = self.wait_for_update('HEARTBEAT', timeout=timeout/5)
                if ret:
                    msg = self.messages['HEARTBEAT']
                else:
                    continue
                if msg is None:
                    continue
            else :
                msg = self.connection.recv_match(type='HEARTBEAT', blocking=True,timeout = timeout)
                if msg is None:
                    continue
            if msg.get_type() != 'HEARTBEAT':
                continue
            msg = msg.to_dict()
            if msg["custom_mode"] == mode_id:
                print(f"Mode {mode} set succesfully")
                # break
                return True
            
    def get_current_flight_mode(self,timeout = 5):
        """
        Returns the current flight mode as a human-readable string.
        """
        # Fetch the latest heartbeat message
        t1 = time.time()
        self.connection.wait_heartbeat()
        while True:
            if self.parralel_telemetry:
                heartbeat = self.messages['HEARTBEAT']
            else :
                heartbeat = self.connection.recv_match(type='HEARTBEAT')
            if heartbeat is not None:
                if heartbeat.type == 1: # 1 for fixed wing
                    break
            if time.time()-t1>timeout:
                print('Error reading fligh mode. Tiemeout')
                return
        # Check the type of the vehicle (e.g., copter, plane, rover)
        #if heartbeat.type == mavutil.mavlink.MAV_TYPE_QUADROTOR:
            # For copters
            #mode_id = heartbeat.custom_mode
            #return mavutil.mode_mapping_acm.get(mode_id, "Unknown")
        #elif heartbeat.type == mavutil.mavlink.MAV_TYPE_FIXED_WING:
            # For planes
        mode_id = heartbeat.custom_mode
        return mavutil.mode_mapping_apm.get(mode_id, "Unknown")

    def set_attitude(self,roll_angle=0, pitch_angle=0, yaw_angle=0, thrust=None):
    ################### ATTITUDE_TARGET_TYPEMASK #################################
    # bits counted from the end
    # 1 (bit0)	ATTITUDE_TARGET_TYPEMASK_BODY_ROLL_RATE_IGNORE	Ignore body roll rate
    # 2	(bit1) ATTITUDE_TARGET_TYPEMASK_BODY_PITCH_RATE_IGNORE	Ignore body pitch rate
    # 4	(bit2) ATTITUDE_TARGET_TYPEMASK_BODY_YAW_RATE_IGNORE	Ignore body yaw rate
    # 32 (bit5)	ATTITUDE_TARGET_TYPEMASK_THRUST_BODY_SET	Use 3D body thrust setpoint instead of throttle
    # 64 (bit6)	ATTITUDE_TARGET_TYPEMASK_THROTTLE_IGNORE	Ignore throttle
    # 128 (bit7) ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE	Ignore attitude
    ##############################################################################
        """ TO DO: define mask for example to ignore thrust"""
        if thrust is None:
            mask = 0b1000000
            thrust = 0
        else:
            mask = 0
        mask = 0
        msg = self.connection.mav.set_attitude_target_encode(
            0,
            self.connection.target_system,
            self.connection.target_component,
            0b0000000, # type mask
            to_quaternion(roll_angle, pitch_angle, yaw_angle), # quaternion
            0, # body roll rate in radian
            0, # body pitch rate in radian
            0, # body yaw rate in radian
            thrust # thrust
        )
        self.connection.mav.send(msg)

    def read_parameter(self, param_id,timeout = 1):
        # Request parameter
        if self.parralel_telemetry:
            self.initiate_wait_for_update('PARAM_VALUE')

        self.connection.mav.param_request_read_send(
            self.connection.target_system, self.connection.target_component,
            param_id.encode(), -1
        )

        t1 = time.time()
        # Listen for the parameter value
        while True:
            if time.time() - t1 > timeout:
                print(f"Failed to receive PARAM_VALUE for {param_id} within the timeout period.")
                return False
            if self.parralel_telemetry:
                ret = self.wait_for_update('PARAM_VALUE', timeout=timeout/5)
                msg = self.messages['PARAM_VALUE']
                if ret:
                    msg = self.messages['PARAM_VALUE']
                else:
                    continue
                if msg is None:
                    continue

            else :
                msg = self.connection.recv_match(type='PARAM_VALUE', blocking=True, timeout=timeout)
                if msg is None:
                    continue
            if msg is not None:
                param_value = msg.param_value
                received_param_id = msg.param_id.strip()  # Removed .decode() here
                self.messages['PARAM_VALUE'] = None
                if received_param_id == param_id:
                    self.messages['PARAM_VALUE'] = None
                    return param_value     

    def set_parameter(self, parameter,value,timeout = 2): 
        t1 = time.time()
        while True:
            self.connection.mav.param_set_send(self.connection.target_system,                 # target_system
                                    self.connection.target_component,                        # target_component
                                    parameter.encode(),                                     # Onboard parameter id, terminated by NULL if the length is less than 16 human-readable chars and WITHOUT null termination (NULL) byte if the length is exactly 16 chars - applications have to provide 16+1 bytes storage if the ID is stored as string
                                    value,                                                # Onboard parameter value
                                    mavutil.mavlink.MAV_PARAM_TYPE_REAL64)              # Onboard parameter type: see the MAV_PARAM_TYPE enum for supported data types
            if self.parralel_telemetry:
                msg = self.messages['PARAM_VALUE']
                if msg is None:
                    continue
            else:
                msg = self.connection.recv_match(type = "PARAM_VALUE",blocking = True, timeout = timeout/5)
                if msg is None:
                    continue
            if msg is None and time.time()-t1>timeout:
                print("Failed to receive PARAM_VALUE within the timeout period.")
                return False
            if msg is not None:
                msg = msg.to_dict()
                if msg["param_id"] == parameter:
                    if msg['param_value'] > value-0.01 and msg['param_value'] < value+0.01:
                        print("parameter set succesfully")
                        self.messages['PARAM_VALUE'] = None
                        return True
                    
    def get_home_global_coordinates(self,timeout = 10):
        # TODO needs to be checked in the field
        """ get home global coordinates. Altitude is in meters relative to mean sea level."""

        if self.parralel_telemetry:
            self.initiate_wait_for_update('HOME_POSITION')

        self.connection.mav.command_long_send(
            self.connection.target_system,
            self.connection.target_component,
            mavutil.mavlink.MAV_CMD_GET_HOME_POSITION,
            0, 0, 0, 0, 0, 0, 0, 0
        )

        # Wait and listen for the HOME_POSITION message
        home_position = None
        t1 = time.time()
        while True:
            if time.time() - t1 > timeout:
                print("Error reciving home position, timeout")
                return False
            if self.parralel_telemetry:
                ret = self.wait_for_update('HOME_POSITION', timeout=timeout/5)
                if ret: 
                    msg = self.messages['HOME_POSITION']
                else:
                    continue
                if msg is None:
                    continue
            else:
                msg = self.connection.recv_match(type='HOME_POSITION', blocking=True,timeout=timeout/5) 
                if msg is None:
                    continue  
            if msg:
                home_position = msg
                return home_position.latitude / 1e7, home_position.longitude / 1e7, home_position.altitude / 1e3
            
    def set_home(self,lat,lon,alt, check = True, timeout = 10):
        # TODO needs to be checked in the field
        print("Setting home ...")
        # t1 = time.time()
        # while True:
        #     if time.time() - t1 > timeout:
        #         print("Error setting home position, timeout")
        #         return False
        message = dialects.MAVLink_command_long_message(target_system=self.connection.target_system,
                                                    target_component=self.connection.target_component,
                                                    command=dialects.MAV_CMD_DO_SET_HOME,
                                                    confirmation=0,
                                                    param1=0,                                   # Set home location: 1=Set home as current location. 0=Use location specified in message parameters. 
                                                    param2=0,                                   # not used
                                                    param3=0,                                   # not used
                                                    param4=0,                                   # not used
                                                    param5=lat,                                   
                                                    param6=lon,                                   
                                                    param7=alt)                                 # altitude relative to mean sea level                           
        
        # send command to the vehicle
        if check and self.parralel_telemetry:
            self.initiate_wait_for_update('COMMAND_ACK')

        self.connection.mav.send(message)

        if check == True:
            r = self.check_command_ack("MAV_CMD_DO_SET_HOME ", dialects.MAV_CMD_DO_SET_HOME, timeout = timeout)
            return r
        else:
            print('Home set (not checked)')
            return True
    
    def set_speed(self,speed,check = False, timeout = 10):
        print("Setting airspeed ...")
        message = dialects.MAVLink_command_long_message(target_system=self.connection.target_system,
                                                    target_component=self.connection.target_component,
                                                    command=dialects.MAV_CMD_DO_CHANGE_SPEED,
                                                    confirmation=0,
                                                    param1=0,                                   # Speed type (0=Airspeed, 1=Ground Speed).
                                                    param2=speed,                               # Target speed (m/s). If airspeed, a value below or above min/max airspeed limits results in no change. a value of -2 uses :ref:`TRIM_ARSPD_CM`
                                                    param3=0,                                   # Throttle as a percentage (0-100%). A value of 0 or negative indicates no change.
                                                    param4=0,                                   # not used
                                                    param5=0,                                   # not used
                                                    param6=0,                                   # not used
                                                    param7=0)                                   # not used
        if check and self.parralel_telemetry:
            self.initiate_wait_for_update('COMMAND_ACK')

        self.connection.mav.send(message)
        if check == False:
            print("Airspeed Set (not checked)")
            return True
        else:
            r = self.check_command_ack("MAV_CMD_DO_CHANGE_SPEED",dialects.MAV_CMD_DO_CHANGE_SPEED, timeout = timeout)
            return  r

    def get_mission_item(self,i,timeout = 1):
        t1 = time.time()
        if self.parralel_telemetry:
            self.initiate_wait_for_update('MISSION_ITEM_INT')
        while True:
            self.connection.mav.mission_request_int_send(self.connection.target_system,
                                            self.connection.target_component,
                                            i)
            # Blocking wait for mission item
            if self.parralel_telemetry:
                ret = self.wait_for_update('MISSION_ITEM_INT', timeout=timeout/5)
                if ret:
                    msg = self.messages['MISSION_ITEM_INT']
                else:
                    continue
                if msg is None:
                    continue
            else:
                msg = self.connection.recv_match(type='MISSION_ITEM_INT',blocking = True, timeout = timeout/5)
            if msg is not None and msg.get_type() == 'MISSION_ITEM_INT':
                if msg.seq == i:
                        return msg
            if time.time()-t1>timeout:
                print('Error reading mission point: Timeout')
                return False
            
    def get_mission_count(self,timeout = 2):
        t1 = time.time()
        if self.parralel_telemetry:
            self.initiate_wait_for_update('MISSION_COUNT')
        while True:
            # Request all mission items
            self.connection.mav.mission_request_list_send(self.connection.target_system, self.connection.target_component)
            # Wait for mission count message
            if self.parralel_telemetry:
                ret = self.wait_for_update('MISSION_COUNT', timeout=timeout/5)
                if ret:
                    message = self.messages['MISSION_COUNT']
                else:
                    continue
                if message is None:
                    continue
            else:
                message = self.connection.recv_match(type='MISSION_COUNT',blocking = True, timeout = timeout)
            if message is not None and message.get_type() == 'MISSION_COUNT':
                count = message.count
                return count
            if time.time()-t1>timeout:
                print('Error reading mission count: Timeout')
                return False
            
    def read_landing_coordinates(self,timeout = 1):
        # frame 0 : MAV_FRAME_GLOBAL	Global (WGS84) coordinate frame + altitude relative to mean sea level (MSL).
        # frame 3 : MAV_FRAME_GLOBAL_RELATIVE_ALT	Global (WGS84) coordinate frame + altitude relative to the home position.
        # frame 10 : MAV_FRAME_GLOBAL_TERRAIN_ALT	Global (WGS84) coordinate frame with AGL altitude (altitude at ground level).

        # Request all mission items
        count = self.get_mission_count(timeout = timeout)
        if count is False:
            print("Bad get_mission_count() value, read_landing_coordinates() failed")
            return False
        # Request each mission item and look for the landing point
        for i in range(count):
            msg = self.get_mission_item(i,timeout = timeout)
            if msg is None:
                print("Bad get_mission_item() value, read_landing_coordinates() failed")
                return False
            if msg.command == mavutil.mavlink.MAV_CMD_NAV_LAND:  # Check if this is a landing waypoint
                lat = msg.x / 1e7  # Convert to decimal degrees
                lon = msg.y / 1e7  # Convert to decimal degrees
                frame = msg.frame
                alt = msg.z
                return lat, lon, alt,frame
        return False
    
    def add_wp_before_landing(self,lat,lon,alt,type = mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,timeout = 1):
        land_mission_item = None
        loc = mavutil.location(lat, lon, alt, 354)
        # Wait for mission count
        mission_count = self.get_mission_count(timeout = timeout)
        # Download each mission item
        mission_items = []
        for i in range(mission_count):
            mission_item = self.get_mission_item(i,timeout = timeout)
            if mission_item.command == 21:
                land_mission_item = mission_item
            else:
                mission_items.append(mission_item)

        if land_mission_item is None:
            print("No landing waypoint found")
            return False

        if self.parralel_telemetry:
            self.initiate_wait_for_update('MISSION_ACK')

        self.connection.mav.mission_clear_all_send(self.connection.target_system, self.connection.target_component)

        # wait for MISSION_ACK 
        if self.parralel_telemetry:
            ret = self.wait_for_update('MISSION_ACK', timeout=timeout)
            if ret == False:
                print("Error recieveing mission ack for mission_clear_all_send")
                return False
        else :
            self.connection.recv_match(type='MISSION_ACK', blocking=True,timeout = timeout)

        self.connection.mav.mission_count_send(self.connection.target_system, self.connection.target_component, len(mission_items)+2)
        for mission_item in mission_items:
            self.connection.mav.mission_item_int_send(
                self.connection.target_system,
                self.connection.target_component,
                mission_item.seq, # seq
                mission_item.frame,
                mission_item.command,
                1, # current - guided-mode request
                1, # autocontinue
                mission_item.param1, # p1
                mission_item.param2, # p2
                mission_item.param3, # p3
                mission_item.param4, # p4
                int(mission_item.x), # latitude
                int(mission_item.y), # longitude
                mission_item.z, # altitude
                mission_item.mission_type)
            # m = connection.recv_match(type='MISSION_ACK', blocking=True, timeout=5)
            # print(m)
            # if m is None:
            #     print("Did not get MISSION_ACK")

        seq = land_mission_item.seq
        self.connection.mav.mission_item_int_send(
            self.connection.target_system,
            self.connection.target_component,
            seq, # seq
            mavutil.mavlink.MAV_FRAME_GLOBAL,
            type,
            1, # current - guided-mode request
            0, # autocontinue
            0, # p1
            0, # p2
            0, # p3
            0, # p4
            int(loc.lat *1e7), # latitude
            int(loc.lng *1e7), # longitude
            loc.alt, # altitude
            mavutil.mavlink.MAV_MISSION_TYPE_MISSION)

        # m = connection.recv_match(type='MISSION_ACK', blocking=True, timeout=5)
        # print(m)
        # if m is None:
        #     print("Did not get MISSION_ACK")
        seq_land = land_mission_item.seq + 1
        self.connection.mav.mission_item_int_send(
            self.connection.target_system,
            self.connection.target_component,
            seq_land, # seq
            land_mission_item.frame,
            land_mission_item.command,
            0, # current - guided-mode request
            1, # autocontinue
            land_mission_item.param1, # p1
            land_mission_item.param2, # p2
            land_mission_item.param3, # p3
            land_mission_item.param4, # p4
            int(land_mission_item.x), # latitude
            int(land_mission_item.y), # longitude
            land_mission_item.z, # altitude
            land_mission_item.mission_type)
        # m = connection.recv_match(type='MISSION_ACK', blocking=True, timeout=5)
        # print(m)
        # if m is None:
        #     print("Did not get MISSION_ACK")

        return seq
        
    def move_land_wp(self,lat,lon,timeout = 1):
        land_mission_item = None
        loc = mavutil.location(lat, lon, 0, 354)
        # Wait for mission count
        mission_count = self.get_mission_count(timeout = timeout)
        # Download each mission item
        mission_items = []
        for i in range(mission_count):
            mission_item = self.get_mission_item(i,timeout = timeout)
            if mission_item.command == 21:
                land_mission_item = mission_item
            else:
                mission_items.append(mission_item)

        if land_mission_item is None:
            print("No landing waypoint found")
            return False

        if self.parralel_telemetry:
            self.initiate_wait_for_update('MISSION_ACK')

        self.connection.mav.mission_clear_all_send(self.connection.target_system, self.connection.target_component)

        # wait for MISSION_ACK 
        if self.parralel_telemetry:
            ret = self.wait_for_update('MISSION_ACK', timeout=timeout)
            if ret == False:
                print("Error recieveing mission ack for mission_clear_all_send")
                return False
        else :
            self.connection.recv_match(type='MISSION_ACK', blocking=True,timeout = timeout)

        self.connection.mav.mission_count_send(self.connection.target_system, self.connection.target_component, len(mission_items)+1)
        for mission_item in mission_items:
            self.connection.mav.mission_item_int_send(
                self.connection.target_system,
                self.connection.target_component,
                mission_item.seq, # seq
                mission_item.frame,
                mission_item.command,
                1, # current - guided-mode request
                1, # autocontinue
                mission_item.param1, # p1
                mission_item.param2, # p2
                mission_item.param3, # p3
                mission_item.param4, # p4
                int(mission_item.x), # latitude
                int(mission_item.y), # longitude
                mission_item.z, # altitude
                mission_item.mission_type)
            # m = connection.recv_match(type='MISSION_ACK', blocking=True, timeout=5)
            # print(m)
            # if m is None:
            #     print("Did not get MISSION_ACK")

        # m = connection.recv_match(type='MISSION_ACK', blocking=True, timeout=5)
        # print(m)
        # if m is None:
        #     print("Did not get MISSION_ACK")
        seq_land = land_mission_item.seq
        self.connection.mav.mission_item_int_send(
            self.connection.target_system,
            self.connection.target_component,
            seq_land, # seq
            land_mission_item.frame,
            land_mission_item.command,
            0, # current - guided-mode request
            1, # autocontinue
            land_mission_item.param1, # p1
            land_mission_item.param2, # p2
            land_mission_item.param3, # p3
            land_mission_item.param4, # p4
            int(loc.lat * 1e7), # latitude
            int(loc.lng * 1e7), # longitude
            land_mission_item.z, # altitude
            land_mission_item.mission_type)
        # m = connection.recv_match(type='MISSION_ACK', blocking=True, timeout=5)
        # print(m)
        # if m is None:
        #     print("Did not get MISSION_ACK")

        return True

    def delate_wp(self,list_of_waypoints_seq_to_delete,timeout = 1):
        # check if list_of_waypoints_seq_to_delete is list type
        if not isinstance(list_of_waypoints_seq_to_delete, list):
            print("list_of_waypoints_seq_to_delete must be a list")
            return False
        try:
            # Wait for mission count
            mission_count = self.get_mission_count()
            # Download each mission item
            mission_items = []
            for i in range(mission_count):
                mission_item = self.get_mission_item(i)
                if mission_item.seq not in list_of_waypoints_seq_to_delete:
                    mission_items.append(mission_item)

            if self.parralel_telemetry:
                self.initiate_wait_for_update('MISSION_ACK')

            self.connection.mav.mission_clear_all_send(self.connection.target_system, self.connection.target_component)
            # wait for MISSION_ACK 
            if self.parralel_telemetry:
                ret = self.wait_for_update('MISSION_ACK', timeout=timeout)
                if ret == False:
                    print("Error recieveing mission ack for mission_clear_all_send")
                    return False
            else :
                self.connection.recv_match(type='MISSION_ACK', blocking=True,timeout = timeout)

            self.connection.mav.mission_count_send(self.connection.target_system, self.connection.target_component, len(mission_items))
            seq = -1
            for mission_item in mission_items:
                seq = seq+1
                self.connection.mav.mission_item_int_send(
                    self.connection.target_system,
                    self.connection.target_component,
                    seq, # seq
                    mission_item.frame,
                    mission_item.command,
                    1, # current - guided-mode request
                    1, # autocontinue
                    mission_item.param1, # p1
                    mission_item.param2, # p2
                    mission_item.param3, # p3
                    mission_item.param4, # p4
                    int(mission_item.x), # latitude
                    int(mission_item.y), # longitude
                    mission_item.z, # altitude
                    mission_item.mission_type)
                # m = connection.recv_match(type='MISSION_ACK', blocking=True, timeout=5)
                # print(m)
                # if m is None:
                #     print("Did not get MISSION_ACK")
            return True
        except:
            return False
    

    def set_current_waypoint(self, waypoint_index,timeout = 1):
        """
        Set the current waypoint to the specified waypoint index.

        Args:
            connection: The pymavlink connection to the vehicle.
            waypoint_index: The index of the waypoint to set as the new current waypoint.

        Returns:
            None
        """
        # Set the current waypoint to waypoint number 3 (for example)
        # Note: Waypoint indices are zero-based in MAVLink
        t1 = time.time()
        if self.parralel_telemetry:
            self.initiate_wait_for_update('MISSION_CURRENT')

        while True:
            if time.time() - t1 > timeout:
                print(f"Failed to set current waypoint to {waypoint_index + 1}. Timeout {timeout} s")
                return False
            waypoint_index = waypoint_index  # Third waypoint, since index is zero-based
            self.connection.mav.mission_set_current_send(
                self.connection.target_system,
                self.connection.target_component,
                waypoint_index
            )
            # Check if the waypoint was set correctly
            if self.parralel_telemetry:
                ret = self.wait_for_update('MISSION_CURRENT', timeout=timeout/5)
                if ret:
                    msg = self.messages['MISSION_CURRENT']
                else:
                    continue
                if msg is None:
                    continue
            else:
                msg = self.connection.recv_match(type='MISSION_CURRENT', blocking=True,timeout = timeout/5)
            if msg is not None:
                current_waypoint = msg.seq
                if current_waypoint == waypoint_index:
                    print(f"Current waypoint set to {waypoint_index + 1}")
                    return True
                else:
                    continue
            return False
        
    def terrain_height(self,lat,lon,timeout = 1):
        lat = int(lat * 1e7)  # Convert latitude to degrees * 1e7
        lon = int(lon * 1e7)  # Convert longitude to degrees * 1e7
        if self.parralel_telemetry:
            self.initiate_wait_for_update('TERRAIN_REPORT')
    
        self.connection.mav.terrain_check_send(lat, lon)
        if self.parralel_telemetry:
            ret = self.wait_for_update('TERRAIN_REPORT', timeout=timeout)
            if ret:
                msg = self.messages['TERRAIN_REPORT']
            else:
                return False
            if msg is None:
                return False
        else:
            msg = self.connection.recv_match(type='TERRAIN_REPORT', blocking=True, timeout=timeout)
        self.messages['TERRAIN_REPORT'] = None
        return msg.terrain_height, msg.current_height
        # print(f"Current waypoint set to {waypoint_index + 1}")

        #elif heartbeat.type == mavutil.mavlink.MAV_TYPE_GROUND_ROVER:
            # For rovers
            #mode_id = heartbeat.custom_mode
            #return mavutil.mode_mapping_rover.get(mode_id, "Unknown")
        #else:
            #return "Unsupported vehicle type"

    # def set_wp_local_through_global_coord(self, x, y, z,timeout = 10,lotier_rad = 80):
    #     self.set_parameter(self.connection,"WP_LOITER_RAD",lotier_rad)
    #     home_lat,home_lan,home_alt = self.get_home_global_coordinates(self.connection)
    #     target_lat, target_lon = add_meters_to_lat_lon(home_lat,home_lan,x,y)

    #     message = dialects.MAVLink_mission_item_int_message(
    #                     target_system=self.connection.target_system,
    #                     target_component=self.connection.target_component,
    #                     seq=0,                                                                  # Waypoint ID (sequence number). Starts at zero. Increases monotonically for each waypoint, no gaps in the sequence (0,1,2,3,4).
    #                     frame=dialects.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,                        # The coordinate system of the waypoint.
    #                     command=dialects.MAV_CMD_NAV_WAYPOINT,                                   # The scheduled action for the waypoint.
    #                     current=2,                                                              # false:0, true:1, guided mode:2
    #                     autocontinue=1,                                                         # Autocontinue to next waypoint. 0: false, 1: true. Set false to pause mission after the item completes.
    #                     param1=0,                                                               # PARAM1 / For NAV command waypoints: Time that the MAV should stay inside the PARAM1 radius before advancing, in milliseconds
    #                     param2=50,                                                               # PARAM2 / For NAV command waypoints: Radius in which the waypoint is accepted as reached, in meters
    #                     param3=0,                                                               # PARAM3 / 0 to pass through the WP, if > 0 radius to pass by WP. Positive value for clockwise orbit, negative value for counter-clockwise orbit. Allows trajectory control.
    #                     param4=0,                                                               # PARAM4 / 	Desired yaw angle at waypoint (rotary wing). NaN to use the current system yaw heading mode (e.g. yaw towards next waypoint, yaw to home, etc.).
    #                     x=int(target_lat * 1e7),                           # PARAM5 / local: x position in meters * 1e4, global: latitude in degrees * 10^7
    #                     y=int(target_lon * 1e7),                          # PARAM6 / local: y position in meters * 1e4, global: longitude in degrees * 10^7
    #                     z=z                                     # PARAM7 / local: z position: positive is down, in meters * 1e4, global: altitude in meters (relative or absolute, depending on frame).
    #                 )

    #     message = dialects.MAVLink_mission_item_int_message(
    #                     target_system=self.connection.target_system,
    #                     target_component=self.connection.target_component,
    #                     seq=1,                                                                  # Waypoint ID (sequence number). Starts at zero. Increases monotonically for each waypoint, no gaps in the sequence (0,1,2,3,4).
    #                     frame=dialects.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,                        # The coordinate system of the waypoint.
    #                     command=dialects.MAV_CMD_NAV_WAYPOINT,                                   # The scheduled action for the waypoint.
    #                     current=2,                                                              # false:0, true:1, guided mode:2
    #                     autocontinue=1,                                                         # Autocontinue to next waypoint. 0: false, 1: true. Set false to pause mission after the item completes.
    #                     param1=0,                                                               # PARAM1 / For NAV command waypoints: Time that the MAV should stay inside the PARAM1 radius before advancing, in milliseconds
    #                     param2=50,                                                               # PARAM2 / For NAV command waypoints: Radius in which the waypoint is accepted as reached, in meters
    #                     param3=0,                                                               # PARAM3 / 0 to pass through the WP, if > 0 radius to pass by WP. Positive value for clockwise orbit, negative value for counter-clockwise orbit. Allows trajectory control.
    #                     param4=0,                                                               # PARAM4 / 	Desired yaw angle at waypoint (rotary wing). NaN to use the current system yaw heading mode (e.g. yaw towards next waypoint, yaw to home, etc.).
    #                     x=int(target_lat * 1e7),                           # PARAM5 / local: x position in meters * 1e4, global: latitude in degrees * 10^7
    #                     y=int(target_lon * 1e7),                          # PARAM6 / local: y position in meters * 1e4, global: longitude in degrees * 10^7
    #                     z=z                                     # PARAM7 / local: z position: positive is down, in meters * 1e4, global: altitude in meters (relative or absolute, depending on frame).
    #                 )

    #     # send target location command to the vehicle
    #     self.connection.mav.send(message)
    #     t1 = time.time()

    #     while True:
    #         if time.time() - t1 > timeout:
    #             print("Error sending message. Timeout")
    #             return False
    #         msg = self.connection.recv_match(type=[dialects.MAVLink_position_target_global_int_message.msgname],
    #                                     blocking=True)
    #         msg = msg.to_dict()
    #         if msg["lat_int"] == int(target_lat * 1e7) and msg["lon_int"] == int(target_lon * 1e7) and int(msg["alt"]) == int(home_alt + z):
    #             print("Target point set successfully")
    #             break

    # def set_wp_global_coord(self, lat, lon, alt, timeout = 10):
    #     # home_lat,home_lan,home_alt = get_home_global_coordinates(connection)
    #     # target_lat, target_lon = add_meters_to_lat_lon(home_lat,home_lan,x,y)
    #     # Request all mission items from the vehicle
    #     self.connection.mav.mission_request_list_send(self.connection.target_system, self.connection.target_component)

    #     # Wait for mission count
    #     mission_count = self.connection.recv_match(type='MISSION_COUNT', blocking=True).count

    #     print(mission_count)
    #     # Download each mission item
    #     # mission_items = []
    #     # for i in range(mission_count):
    #     #     vehicle.mav.mission_request_int_send(vehicle.target_system, vehicle.target_component, i)
    #     #     mission_item = vehicle.recv_match(type='MISSION_ITEM_INT', blocking=True)
    #     #     mission_items.append(mission_item)

        
    #     print(f'setting waypoint: lat={lat}, lon={lon}, alt={alt}')
    #     message = dialects.MAVLink_mission_item_int_message(
    #                     target_system=self.connection.target_system,
    #                     target_component=self.connection.target_component,
    #                     seq=mission_count,                                                                  # Waypoint ID (sequence number). Starts at zero. Increases monotonically for each waypoint, no gaps in the sequence (0,1,2,3,4).
    #                     frame=dialects.MAV_FRAME_GLOBAL,                                    # The coordinate system of the waypoint. Altitude is relative to mean sea level 
    #                     command=dialects.MAV_CMD_NAV_WAYPOINT,                                   # The scheduled action for the waypoint.
    #                     current=2,                                                              # false:0, true:1, guided mode:2
    #                     autocontinue=0,                                                         # Autocontinue to next waypoint. 0: false, 1: true. Set false to pause mission after the item completes.
    #                     param1=0,                                                               # PARAM1 / For NAV command waypoints: Time that the MAV should stay inside the PARAM1 radius before advancing, in milliseconds
    #                     param2=1,                                                               # PARAM2 / For NAV command waypoints: Radius in which the waypoint is accepted as reached, in meters
    #                     param3=0,                                                               # PARAM3 / 0 to pass through the WP, if > 0 radius to pass by WP. Positive value for clockwise orbit, negative value for counter-clockwise orbit. Allows trajectory control.
    #                     param4=0,                                                               # PARAM4 / 	Desired yaw angle at waypoint (rotary wing). NaN to use the current system yaw heading mode (e.g. yaw towards next waypoint, yaw to home, etc.).
    #                     x=int(lat * 1e7),                           # PARAM5 / local: x position in meters * 1e4, global: latitude in degrees * 10^7
    #                     y=int(lon * 1e7),                          # PARAM6 / local: y position in meters * 1e4, global: longitude in degrees * 10^7
    #                     z=alt                                     # PARAM7 / local: z position: positive is down, in meters * 1e4, global: altitude in meters (relative or absolute, depending on frame).
    #                 )

    #     # send target location command to the vehicle
    #     self.connection.mav.send(message)
    #     # ack_msg = connection.recv_match(type='MISSION_ACK', blocking=True, timeout=10)
    #     # print(ack_msg)
    #     t1 = time.time()
    #     while True:
    #         if time.time() - t1 > timeout:
    #             print("Error setting waypoint: ACK message not recieved. Timeout 10s")
    #             return False
    #         msg = self.connection.recv_match(type=[dialects.MAVLink_position_target_global_int_message.msgname],
    #                                     blocking=True,timeout = timeout)
    #         if msg is None:
    #             print("Failed to receive position_target_global_int_message within the timeout period.")
    #             return False
    #         msg = msg.to_dict()
    #         if msg["lat_int"] == int(lat * 1e7) and msg["lon_int"] == int(lon * 1e7) and int(msg["alt"]) == int(alt):
    #             print("Target point set successfully")
    #             break
            


# def main():
#     connection_string = 'COM13'  
#     # connection_string = "udpin:localhost:14551"
#     mavlink_wrapper = MavlinkWrapper('COM13')
#     mavlink_wrapper.connect()
#     mavlink_wrapper.run_telemetry()
#     # mavlink_wrapper.set_message_rate(mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, 1)
#     # time.sleep(3)
#     # t1 = time.time()
#     while True:
#         mavlink_wrapper.set_rc_channel_pwm(channel_id = 1, pwm=1100)
#         read_rc_channel = mavlink_wrapper.read_rc_channel(1)
#         print(read_rc_channel)

if __name__ == "__main__":
    connection_string = 'COM13'  
    # connection_string = "udpin:localhost:14551"
    source_system = 255
    mavlink_wrapper = MavlinkWrapper(connection_string, source_system=source_system)
    mavlink_wrapper.connect()
    mavlink_wrapper.set_mode('FBWA') 
    # mavlink_wrapper.run_telemetry_parralel()
    mavlink_wrapper.set_message_rate(mavutil.mavlink.MAVLINK_MSG_ID_RC_CHANNELS, 1)
    mavlink_wrapper.set_message_rate(mavutil.mavlink.MAVLINK_MSG_ID_SERVO_OUTPUT_RAW, 1)
    # mavlink_wrapper.set_mode('MANUAL')
    mavlink_wrapper.connection.mav.rc_channels_override_send(
        mavlink_wrapper.connection.target_system,
        mavlink_wrapper.connection.target_component,
        1500, 1500, 1300, 1500, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0  # Fill rest with zeros
    )
    while True:
        mavlink_wrapper.set_message_rate(mavutil.mavlink.MAVLINK_MSG_ID_SERVO_OUTPUT_RAW, 1)
        mavlink_wrapper.set_rc_channel_pwm(2,1300)
        msg = mavlink_wrapper.connection.recv_match(type='RC_CHANNELS', blocking=True, timeout=0.1)
        if msg:
            print(f"RC1={msg.chan1_raw}; RC2={msg.chan2_raw}; RC3={msg.chan3_raw}; RC3={msg.chan4_raw}")
        time.sleep(0.1)