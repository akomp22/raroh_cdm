import numpy as np

class PIDFFController:
    def __init__(self, Kp, Ki, Kd, Kff, i_max, min_cmd, max_cmd,
                 error_alpha=0.9, derivative_alpha=0.8, max_history=1000):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.Kff = Kff
        self.i_max = i_max
        self.min_cmd = min_cmd
        self.max_cmd = max_cmd
        self.error_alpha = error_alpha
        self.derivative_alpha = derivative_alpha
        self.max_history = max_history

        # Buffers and time
        self.errors = []
        self.dts = []
        self.prev_time = None

        # Filtering state
        self.filtered_error = None
        self.prev_filtered_error = None
        self.filtered_derivative = 0.0

    def get_command(self, setpoint, current_value, current_time):
        error = setpoint - current_value

        # Time difference
        if self.prev_time is None:
            dt = 0.0
        else:
            dt = current_time - self.prev_time
            dt = max(dt, 1e-6)  # Prevent divide-by-zero

        self.prev_time = current_time
        self.errors.append(error)
        if dt > 0:
            self.dts.append(dt)

        # Trim history
        if len(self.errors) > self.max_history:
            self.errors.pop(0)
        if len(self.dts) > self.max_history:
            self.dts.pop(0)

        # === Filter the error signal ===
        if self.filtered_error is None:
            self.filtered_error = error
            self.prev_filtered_error = error
            raw_derivative = 0.0  # no derivative on first run
        else:
            self.filtered_error = (self.error_alpha * self.filtered_error +
                                (1 - self.error_alpha) * error)
            raw_derivative = (self.filtered_error - self.prev_filtered_error) / dt
            self.prev_filtered_error = self.filtered_error

        # === Filter the derivative ===
        self.filtered_derivative = (self.derivative_alpha * self.filtered_derivative +
                                    (1 - self.derivative_alpha) * raw_derivative)

        # === Integral calculation using raw error ===
        if len(self.dts) > 0:
            integral = np.sum(np.array(self.errors[1:]) * np.array(self.dts))
            integral = np.clip(integral, -self.i_max, self.i_max)
        else:
            integral = 0.0

        # === Final command ===
        cmd = (self.Kp * self.filtered_error +
            self.Ki * integral +
            self.Kd * self.filtered_derivative +
            self.Kff * setpoint)

        return np.clip(cmd, self.min_cmd, self.max_cmd)


    def reset(self):
        self.errors.clear()
        self.dts.clear()
        self.prev_time = None
        self.filtered_error = None
        self.prev_filtered_error = None
        self.filtered_derivative = 0.0
