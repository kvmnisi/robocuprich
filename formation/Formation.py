import numpy as np

def GenerateBasicFormation(ball_pos = None):


    #  formation = [
    #     np.array([-13, 0]),    # Goalkeeper
    #      np.array([-7, -2]),  # Left Defender
    #      np.array([-0, 3]),   # Right Defender
    #      np.array([7, 1]),    # Forward Left
    # #     np.array([12, 0])      # Forward Right
    # # ]
    formation = [
        np.array([-13, 0]),    # Goalkeeper
         np.array([-3, 0]),  # CB
         np.array([8, -8]),   # Right Defender
         np.array([8, 8]),    # Forward Left
         np.array([12, 0])      # Forward Right
     ]
    
    # return {
    #     1: (-13, 0),     # Goalkeeper - stays back
    #     2: (-6, -3),     # Left defender/midfielder
    #     3: (-2, -2),     # Left midfielder  
    #     4: (-2, 2),      # Right midfielder
    #     5: (2, 0)        # Forward - central
    # }

    # formation = [
    #     np.array([-13, 0]),    # Goalkeeper
    #     np.array([-10, -2]),  # Left Defender
    #     np.array([-11, 3]),   # Center Back Left
    #     np.array([-8, 0]),    # Center Back Right
    #     np.array([-3, 0]),   # Right Defender
    #     np.array([0, 1]),    # Left Midfielder
    #     np.array([2, 0]),    # Center Midfielder Left
    #     np.array([3, 3]),     # Center Midfielder Right
    #     np.array([8, 0]),     # Right Midfielder
    #     np.array([9, 1]),    # Forward Left
    #     np.array([12, 0])      # Forward Right
    # ]
    diski = ball_pos[0]
    if -15<= diski < - 5:
        shift_x = -3.0
        width = 0.3
    elif -5 <= diski <=5:
        shift_x = -1.5
        width = 1.0
    else:
        shift_x = 0
        width = 0.7
    
    for i in range(2, len(formation)):
        formation[i][0] += shift_x
        formation[i][1] *= width
    return formation


def KickOffFormation():
    formation =  [
        np.array([-13, 0]),    # Goalkeeper
         np.array([-6, -3]),  # Left Defender
         np.array([-2, -2]),   # Right Defender
         np.array([-2, 2]),    # Forward Left
         np.array([-3, 0])      # Forward Right
     ]
    
    return formation