"""utils_eyetracking.py
Utility functions to integrating eyetracking devices to experiments.
"""
import numpy as np
import threading
from subprocess import Popen, PIPE

class RandomGazeGen(object):
    """Generates random normalized gaze data while not eye tracking hardware
    is not integrated to AirSim.
    """
    def __init__(self):
        # initial gaze values
        self.gaze_x = np.random.rand()
        self.gaze_y = np.random.rand()

    def read_gaze(self):
        # add random noise to previous gaze reading
        self.gaze_x += (np.random.rand()-0.5)/10
        self.gaze_y += (np.random.rand()-0.5)/10

        # make sure it is still in the image
        self.gaze_x = np.clip(self.gaze_x, 0, 1)
        self.gaze_y = np.clip(self.gaze_y, 0, 1)

        return [self.gaze_x, self.gaze_y]


class Tobii4C_Cpp(object):
    """Pipes gaze data from Tobii 4C written in C++.
    """
    def __init__(self):
        # initialize pipe from Cpp to Python
        self.cpp_pipe = Popen(['../eye_tracking/start_eyetracker.sh'],
                       shell=True, stdout=PIPE, stdin=PIPE)

        # initial gaze values
        self.gaze_x = 0.
        self.gaze_y = 0.
        self.running_eyetracker = True

        # update gaze values in separate thread
        self.gaze_thread = threading.Thread(target=self._update_gaze_values)
        self.gaze_thread.start()

    def _update_gaze_values(self):
        while self.running_eyetracker:
            # read data coming from cpp
            result_stdout = str(
                self.cpp_pipe.stdout.readline().strip()).split(',')

            # fix formatting and display
            self.gaze_x = float(result_stdout[0][2:])
            self.gaze_y = float(result_stdout[1][:-1])

    def read_gaze(self):
        return [self.gaze_x, self.gaze_y]

    def close(self):
        # stops eye tracker thread
        self.running_eyetracker = False
        self.gaze_thread.join()