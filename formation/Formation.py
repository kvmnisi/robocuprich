import numpy as np

def GenerateBasicFormation(ball_pos = None):
    formation = [
        np.array([-13, 0]),    # Goalkeeper
         np.array([-3, 0]),  # CB
         np.array([8, 3]),   # LM
         np.array([8, -2]),    #  RM
         np.array([12, 0])      # ST
     ]
    return formation


def KickOffFormation():
    formation = [
        np.array([-13, 0]),    # Goalkeeper
        np.array([-1, 0]),  # CB
        np.array([-2, 5]),   # Right Defender
        np.array([-1, -4]),    # Forward Left
        np.array([-2, -1])      # Forward Right
            ]
    
    return formation

def Vala(ball_pos = None):
    formation = [
        np.array([-13, 0]),    # Goalkeeper
        np.array([-8, -2]),  # CB
        np.array([-8, 2]),   # Right Defender
        np.array([-4, 0]),    # Forward Left
        np.array([0, 0])      # Forward Right
            ]
    
    return formation