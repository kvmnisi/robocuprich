import numpy as np

def GenerateBasicFormation(ball_pos = None):
    formation = [
        np.array([-13, 0]),    # Goalkeeper
         np.array([-3, 0]),  # CB
         np.array([8, 3]),   # rm
         np.array([8, -2]),    #  LM
         np.array([12, 0])      # ST
     ]
    
    # if ball_pos is None:
    #     diski = 0
    # else:
    #     diski = ball_pos[0]
    # zonefactor = np.clip(diski / 15.0, -1.0, 1.0)
    # shift_x = 2.0 * zonefactor
    # width = 1.4-0.4*zonefactor
    # for i in range(2, len(formation)):
    #     formation[i][0] += shift_x
    #     formation[i][1] *= width
    return formation


def KickOffFormation():
    formation = [
        np.array([-13, 0]),    # Goalkeeper
        np.array([-5, 0]),  # CB
        np.array([-2, 5]),   # Right Defender
        np.array([-2, -4]),    # Forward Left
        np.array([-1, 0])      # Forward Right
            ]
    
    return formation