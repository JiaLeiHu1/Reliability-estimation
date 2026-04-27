#!/usr/bin/env python
import math

DELTA_TIMESTAMP_MS = 100  # similar throughout the whole dataset

class MotionState:
    def __init__(self, time_stamp_ms):
        assert isinstance(time_stamp_ms, int)
        self.time_stamp_ms = time_stamp_ms
        self.x = None
        self.y = None
        self.vx = None
        self.vy = None
        self.psi_rad = None
        #暂定轿车质量为1500kg
        self.mass = None
    def update_mass(self):
        assert self.vx != None and self.vy != None
        self.mass = 150000*(1.566*10**(-14)*math.sqrt((self.vx)**2)+(self.vy)**2)

    def __str__(self):
        return "MotionState: " + str(self.__dict__)


class Track:
    def __init__(self, id):
        # assert isinstance(id, int)
        self.track_id = id
        self.agent_type = None
        self.length = None
        self.width = None
        self.time_stamp_ms_first = None
        self.time_stamp_ms_last = None
        self.motion_states = dict()

    def __str__(self):
        string = "Track: track_id=" + str(self.track_id) + ", agent_type=" + str(self.agent_type) + \
                 ", length=" + str(self.length) + ", width=" + str(self.width) + \
                 ", time_stamp_ms_first=" + str(self.time_stamp_ms_first) + \
                 ", time_stamp_ms_last=" + str(self.time_stamp_ms_last) + \
                 "\n motion_states:"
        for key, value in sorted(self.motion_states.items()):
            string += "\n    " + str(key) + ": " + str(value)
        return string
