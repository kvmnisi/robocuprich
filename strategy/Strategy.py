import numpy as np
import math
from math_ops.Math_Ops import Math_Ops as M

class Strategy():
    def __init__(self, world):
        self.world = world
        self.play_mode = world.play_mode
        self.robot_model = world.robot  
        self.my_head_pos_2d = self.robot_model.loc_head_position[:2]
        self.player_unum = self.robot_model.unum
        self.mypos = (world.teammates[self.player_unum-1].state_abs_pos[0],
                      world.teammates[self.player_unum-1].state_abs_pos[1])
       
        self.side = 1
        if world.team_side_is_left:
            self.side = 0

        self.teammate_positions = [teammate.state_abs_pos[:2] if teammate.state_abs_pos is not None 
                                    else None
                                    for teammate in world.teammates
                                    ]
        
        self.opponent_positions = [opponent.state_abs_pos[:2] if opponent.state_abs_pos is not None 
                                    else None
                                    for opponent in world.opponents
                                    ]

        self.valid_opponent_positions = [pos for pos in self.opponent_positions if pos is not None]
        self.valid_teammate_positions = [pos for pos in self.teammate_positions if pos is not None]

        self.my_ori = self.robot_model.imu_torso_orientation
        self.ball_2d = world.ball_abs_pos[:2]
        self.ball_abs_pos = world.ball_abs_pos
        self.ball_vec = self.ball_2d - self.my_head_pos_2d
        self.ball_dir = M.vector_angle(self.ball_vec)
        self.ball_dist = np.linalg.norm(self.ball_vec)
        self.ball_sq_dist = self.ball_dist * self.ball_dist
        self.ball_speed = np.linalg.norm(world.get_ball_abs_vel(6)[:2])
        
        self.goal_dir = M.target_abs_angle(self.ball_2d,(15.05,0))
        self.PM_GROUP = world.play_mode_group

        self.slow_ball_pos = world.get_predicted_ball_pos(0.5)

        self.teammates_ball_sq_dist = [np.sum((p.state_abs_pos[:2] - self.slow_ball_pos) ** 2)
                                  if p.state_last_update != 0 and (world.time_local_ms - p.state_last_update <= 360 or p.is_self) and not p.state_fallen
                                  else 1000
                                  for p in world.teammates ]

        self.opponents_ball_sq_dist = [np.sum((p.state_abs_pos[:2] - self.slow_ball_pos) ** 2)
                                  if p.state_last_update != 0 and world.time_local_ms - p.state_last_update <= 360 and not p.state_fallen
                                  else 1000
                                  for p in world.opponents ]

        self.min_teammate_ball_sq_dist = min(self.teammates_ball_sq_dist)
        self.min_teammate_ball_dist = math.sqrt(self.min_teammate_ball_sq_dist)
        self.min_opponent_ball_dist = math.sqrt(min(self.opponents_ball_sq_dist))

        self.active_player_unum = self.teammates_ball_sq_dist.index(self.min_teammate_ball_sq_dist) + 1

        self._distance_cache = {}


    # ============================================
    # BASIC HELPERS
    # ============================================
    def IsFormationReady(self, point_preferences):
        
        is_formation_ready = True
        for i in range(1, 6):
            if i != self.active_player_unum: 
                teammate_pos = self.teammate_positions[i-1]

                if not teammate_pos is None:

                    distance = np.sum((teammate_pos - point_preferences[i]) **2)
                    if(distance > 0.3):
                        is_formation_ready = False

        return is_formation_ready

    def GetDirectionRelativeToMyPositionAndTarget(self,target):
        target_vec = target - self.my_head_pos_2d
        target_dir = M.vector_angle(target_vec)

        return target_dir
    def distance(self, pos1, pos2):
        """Calculate Euclidean distance"""
        if pos1 is None or pos2 is None:
            return float('inf')
        
        if len(pos1) > 2:
            pos1 = pos1[:2]
        if len(pos2) > 2:
            pos2 = pos2[:2]
        
        cache_key = (round(pos1[0], 2), round(pos1[1], 2), 
                     round(pos2[0], 2), round(pos2[1], 2))
        
        if cache_key in self._distance_cache:
            return self._distance_cache[cache_key]
        
        dist = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        self._distance_cache[cache_key] = dist
        return dist
    
    def distance_squared(self, pos1, pos2):
        """Calculate squared distance"""
        if pos1 is None or pos2 is None:
            return float('inf')
        if len(pos1) > 2:
            pos1 = pos1[:2]
        if len(pos2) > 2:
            pos2 = pos2[:2]
        return (pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2

    def am_i_closest_to_ball(self):
        """Check if I'm closest to ball"""
        return self.active_player_unum == self.player_unum
    
    def can_i_kick(self):
        """Check if close enough to kick"""
        KICK_DISTANCE_SQ = 0.25
        return self.ball_sq_dist < KICK_DISTANCE_SQ


    # ============================================
    # TIKI-TAKA: DYNAMIC FORMATION
    # ============================================
    
    def calculate_tiki_taka_position(self, base_formation_list, my_unum):
        """
        Dynamic tiki-taka positioning for 5v5.
        - Ensures passing options ahead of ball
        - Encourages overlap (pass-and-go) behavior
        - Compact and adaptive to ball movement
        """

        ball = np.array(self.ball_2d)
        goal = np.array((15, 0))
        my_base_pos = np.array(base_formation_list[my_unum - 1])

        # --- Determine relative role ---
        ball_carrier_unum = self.active_player_unum
        am_i_carrier = (my_unum == ball_carrier_unum)

        # Distance from ball
        dist_to_ball = self.distance(my_base_pos, ball)

        # --- Base following logic ---
        # All players follow the ball partially
        follow_factor_x = 0.3
        follow_factor_y = 0.5
        new_pos = np.array([
            my_base_pos[0] + ball[0] * follow_factor_x,
            my_base_pos[1] + ball[1] * follow_factor_y
        ])

        # --- Adjust by role ---
        if am_i_carrier:
            # BALL CARRIER stays just behind ball (so they can pass forward)
            approach_offset = -0.5
            direction_to_goal = (goal - ball)
            direction_to_goal /= (np.linalg.norm(direction_to_goal) + 1e-5)
            new_pos = ball + direction_to_goal * approach_offset

        else:
            # --- If not the carrier, decide based on geometry relative to ball ---
            carrier_pos = np.array(self.teammate_positions[ball_carrier_unum - 1])

            if carrier_pos is not None:
                # Vector from carrier to goal
                carrier_to_goal = goal - carrier_pos
                carrier_to_goal /= (np.linalg.norm(carrier_to_goal) + 1e-5)

                # Vector from carrier to me
                rel_to_carrier = new_pos - carrier_pos
                rel_dist = np.linalg.norm(rel_to_carrier)

                # --- Case 1: I'm very close to the carrier (likely just passed) ---
                if rel_dist < 2.0 and carrier_pos[0] < ball[0]:
                    # Make an overlapping forward run toward goal
                    overlap_distance = 8.0
                    new_pos = carrier_pos + carrier_to_goal * overlap_distance

                # --- Case 2: I’m a nearby support option (within 6m) ---
                elif rel_dist < 6.0:
                    # Stay slightly diagonal, offering lateral pass
                    lateral_offset = np.array([-carrier_to_goal[1], carrier_to_goal[0]]) * 1.5
                    support_offset = carrier_to_goal * 1.5
                    new_pos = carrier_pos + support_offset + lateral_offset

                # --- Case 3: I’m far from ball (defensive fallback) ---
                else:
                    # Maintain base position but slightly move toward ball
                    new_pos = my_base_pos * 0.7 + ball * 0.3

        # --- Clamp boundaries ---
        new_pos[0] = np.clip(new_pos[0], -14.5, 14.5)
        new_pos[1] = np.clip(new_pos[1], -9.5, 9.5)
        if my_unum == 1:
            new_pos[0] = min(new_pos[0], -6.0)
            new_pos[1] = np.clip(new_pos[1], -3.0, 3.0)


        # --- Rule: Defensive players stay behind ball ---
        if my_base_pos[0] < 0:
            new_pos[0] = min(new_pos[0], ball[0] - 1.0)

        return tuple(new_pos)



    # ============================================
    # PASSING LOGIC
    # ============================================
    
    def find_best_pass_target(self):
        """
        Find best teammate to pass to
        
        Returns:
            tuple: (teammate_pos, score) or (None, 0) if no good pass
        """
        best_target = None
        best_score = -999
        opponent_goal = (15, 0)
        
        for i, teammate_pos in enumerate(self.teammate_positions):
            # Skip self and None positions
            if teammate_pos is None or i == self.player_unum - 1:
                continue
            
            # Calculate pass distance
            pass_dist = self.distance(self.ball_2d, teammate_pos)
            
            # Skip if too close or too far
            if pass_dist < 1.0:
                continue
            
            # Check if lane is blocked
            if self.is_passing_lane_blocked(self.ball_2d, teammate_pos, safety_radius=0.6):
                continue
            
            # Calculate score
            score = 0
            
            # Prefer forward passes
            forward_progress = teammate_pos[0] - self.ball_2d[0]
            if forward_progress > 0:
                score += forward_progress * 30
            else:
                score += forward_progress * 5  # Small penalty for backward
            
            # Prefer closer to goal
            dist_to_goal = self.distance(teammate_pos, opponent_goal)
            score += (30 - dist_to_goal) * 3
            
            # Prefer shorter passes (tiki-taka style)
            if pass_dist <= 7.0:
                score += 40
            elif pass_dist <= 6.0:
                score += 20
            
            if score > best_score:
                best_score = score
                best_target = teammate_pos
        
        return best_target, best_score


    def should_shoot(self):
        """
        Decide if we should shoot at goal
        
        Returns:
            bool: True if should shoot
        """
        opponent_goal = (15, 0)
        dist_to_goal = self.distance(self.ball_2d, opponent_goal)
        
        # Only shoot if close enough
        if dist_to_goal > 10:
            return False
        
        # Shoot if very close
        if dist_to_goal < 7.0:
            return True
        
        # Shoot if decent angle and not blocked
        if dist_to_goal < 8.0 and abs(self.ball_2d[1]) < 4.0:
            # Check if shooting lane is clear
            opponents_blocking = 0
            for opp_pos in self.valid_opponent_positions:
                dist_to_line = self.point_to_line_segment_distance(
                    opp_pos, self.ball_2d, opponent_goal
                )
                if dist_to_line < 1.0:
                    opponents_blocking += 1
            
            return opponents_blocking <= 1
        
        return False


    # ============================================
    # GEOMETRY HELPERS
    # ============================================
    
    def point_to_line_segment_distance(self, point, line_start, line_end):
        """Calculate minimum distance from point to line segment"""
        if point is None or line_start is None or line_end is None:
            return float('inf')
        
        P = np.array(point[:2])
        A = np.array(line_start[:2])
        B = np.array(line_end[:2])
        
        AB = B - A
        AP = P - A
        
        ab_length_sq = np.dot(AB, AB)
        if ab_length_sq == 0:
            return np.linalg.norm(AP)
        
        t = np.dot(AP, AB) / ab_length_sq
        t = max(0, min(1, t))
        
        closest_point = A + t * AB
        return np.linalg.norm(P - closest_point)
    
    
    def is_passing_lane_blocked(self, start_pos, end_pos, safety_radius=0.6):
        """Check if passing lane is blocked by opponents"""
        for opp_pos in self.valid_opponent_positions:
            dist_to_line = self.point_to_line_segment_distance(opp_pos, start_pos, end_pos)
            if dist_to_line < safety_radius:
                return True
        return False
    
    
    def get_closest_opponent_to_ball(self):
        """Find closest opponent to ball"""
        if not self.opponents_ball_sq_dist:
            return None, float('inf')
        
        min_sq_dist = min(self.opponents_ball_sq_dist)
        min_index = self.opponents_ball_sq_dist.index(min_sq_dist)
        
        closest_opp = self.opponent_positions[min_index]
        distance = math.sqrt(min_sq_dist)
        
        return closest_opp, distance


    def GetDirectionRelativeToMyPositionAndTarget(self, target):
        """Get direction to target"""
        target_vec = target - self.my_head_pos_2d
        target_dir = M.vector_angle(target_vec)
        return target_dir