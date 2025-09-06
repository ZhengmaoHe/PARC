"""
A kinematic motion generator class.
Generates a variable length sequence of motions, optionally conditioned on:
 - previous N character states
 - goal/objective
 - action ID
 - observations
"""

import abc
import pickle

class MotionGenerator:

    def __init__(self, cfg):
        self._device = cfg['device']

        return
    
    @abc.abstractmethod
    def gen_sequence(self, cond):
        """
        cond: a dictionary of conditions
        """
        motion_seq = None
        info = None
        return motion_seq, info
    
    @abc.abstractmethod
    def sequence_length(self):
        return
    
    def save(self, save_location):
        with open(save_location, "wb") as output_file:
            pickle.dump(self, output_file)
            print("saved to ", save_location)
        return