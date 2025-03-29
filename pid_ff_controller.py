import numpy as np

class PID_FF_controller():
    def __init__(self,Kp,Ki,Kd, Kff, i_max, min_cmd, max_cmd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.Kff = Kff
        self.i_max = i_max
        self.errors = []
        self.dts = []
        self.prev_time = None
        self.min_cmd = min_cmd
        self.max_cmd = max_cmd

    def get_command(self,setpoint,current_value,current_time):
        P = 0
        integral = 0
        D = 0
        error = setpoint - current_value
        self.errors.append(error)
        if self.prev_time is None:
            self.prev_time = current_time
        else:
            dt = current_time - self.prev_time
            # if dt == 0:
            #     dt = 0.01
            self.dts.append(dt)
            integral = np.sum(np.array(self.errors[1:])*np.array(self.dts))
            integral = np.min([abs(integral),self.i_max])*np.sign(integral)
        if len(self.errors) > 10000:
            self.errors.pop(0)
            self.dts.pop(0)
        if len(self.errors) > 2:
            differential = (self.errors[-1]- self.errors[-2])/dt
        else :
            differential = 0
        cmd = self.Kp * error + self.Ki * integral + self.Kd * differential + self.Kff*setpoint
        self.prev_time = current_time
        if cmd < self.min_cmd:
            cmd = self.min_cmd
        if cmd > self.max_cmd:
            cmd = self.max_cmd
        return cmd 
    
    def reset(self):
        self.errors = []
        self.dts = []
        self.prev_time = None
    