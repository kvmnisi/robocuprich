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

        # Filter out None values
        self.valid_opponent_positions = [pos for pos in self.opponent_positions if pos is not None]
        self.valid_teammate_positions = [pos for pos in self.teammate_positions if pos is not None]

        self.team_dist_to_ball = None
        self.team_dist_to_oppGoal = None
        self.opp_dist_to_ball = None

        self.prev_important_positions_and_values = None
        self.curr_important_positions_and_values = None
        self.point_preferences = None
        self.combined_threat_and_definedPositions = None

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

        self.my_desired_position = self.mypos
        self.my_desired_orientation = self.ball_dir

        # Cache for expensive calculations
        self._distance_cache = {}


    # ============================================
    # DISTANCE HELPERS
    # ============================================
    
    def distance(self, pos1, pos2):
        """Calculate Euclidean distance between two positions"""
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
        """Calculate squared distance (faster for comparisons)"""
        if pos1 is None or pos2 is None:
            return float('inf')
        
        if len(pos1) > 2:
            pos1 = pos1[:2]
        if len(pos2) > 2:
            pos2 = pos2[:2]
        
        return (pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2


    # ============================================
    # BALL POSSESSION HELPERS
    # ============================================
    
    def am_i_closest_to_ball(self):
        """Check if I'm the closest teammate to the ball"""
        return self.active_player_unum == self.player_unum
    
    def can_i_kick(self):
        """Check if I'm close enough to kick the ball"""
        KICK_DISTANCE_SQ = 0.25  # 0.5m squared
        return self.ball_sq_dist < KICK_DISTANCE_SQ


    # ============================================
    # OPPONENT AWARENESS
    # ============================================
    
    def get_closest_opponent_to_ball(self):
        """Find the closest opponent to the ball"""
        if not self.opponents_ball_sq_dist:
            return None, float('inf')
        
        min_sq_dist = min(self.opponents_ball_sq_dist)
        min_index = self.opponents_ball_sq_dist.index(min_sq_dist)
        
        closest_opp = self.opponent_positions[min_index]
        distance = math.sqrt(min_sq_dist)
        
        return closest_opp, distance
    
    def get_closest_opponent_to_position(self, position):
        """Find closest opponent to a given position"""
        if not self.valid_opponent_positions:
            return None, float('inf')
        
        min_dist = float('inf')
        closest_opp = None
        
        for opp_pos in self.valid_opponent_positions:
            dist = self.distance(position, opp_pos)
            if dist < min_dist:
                min_dist = dist
                closest_opp = opp_pos
        
        return closest_opp, min_dist
    
    def is_opponent_nearby(self, position, radius=2.0):
        """Check if any opponent is within radius of position"""
        radius_sq = radius * radius
        for opp_pos in self.valid_opponent_positions:
            if self.distance_squared(position, opp_pos) < radius_sq:
                return True
        return False


    # ============================================
    # FIELD POSITION HELPERS
    # ============================================
    
    def is_ball_in_my_half(self):
        """Check if ball is in my defensive half"""
        return self.ball_2d[0] < 0
    
    def is_ball_dangerous(self):
        """Check if ball is dangerously close to my goal"""
        return self.ball_2d[0] < -10
    
    def is_ball_near_goal(self, goal_x=15, threshold=8):
        """Check if ball is near opponent's goal"""
        return self.distance(self.ball_2d, (goal_x, 0)) < threshold


    # ============================================
    # PASSING & GEOMETRY HELPERS
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
    
    def is_passing_lane_blocked(self, start_pos, end_pos, safety_radius=0.5):
        """Check if a passing lane is blocked by opponents"""
        for opp_pos in self.valid_opponent_positions:
            dist_to_line = self.point_to_line_segment_distance(opp_pos, start_pos, end_pos)
            if dist_to_line < safety_radius:
                return True
        return False
    
    def find_clear_passing_targets(self, max_distance=10, min_forward_progress=2):
        """Find all teammates with clear passing lanes"""
        clear_targets = []
        opponent_goal = (15, 0)
        
        for i, teammate_pos in enumerate(self.valid_teammate_positions):
            unum = i + 1
            
            if unum == self.player_unum:
                continue
            
            pass_dist = self.distance(self.mypos, teammate_pos)
            if pass_dist > max_distance or pass_dist < 1:
                continue
            
            forward_progress = teammate_pos[0] - self.ball_2d[0]
            if forward_progress < min_forward_progress:
                continue
            
            if not self.is_passing_lane_blocked(self.mypos, teammate_pos):
                dist_to_goal = self.distance(teammate_pos, opponent_goal)
                score = (30 - dist_to_goal) + forward_progress
                clear_targets.append((unum, teammate_pos, score))
        
        clear_targets.sort(key=lambda x: x[2], reverse=True)
        return clear_targets


    # ============================================
    # STRATEGIC DECISIONS
    # ============================================
    
    def should_i_shoot(self, shooting_range=8):
        """Decide if I should shoot at goal"""
        opponent_goal = (15, 0)
        
        if self.distance(self.ball_2d, opponent_goal) > shooting_range:
            return False
        
        opponents_in_lane = 0
        for opp_pos in self.valid_opponent_positions:
            dist_to_shooting_line = self.point_to_line_segment_distance(
                opp_pos, self.ball_2d, opponent_goal
            )
            if dist_to_shooting_line < 1.5:
                opponents_in_lane += 1
        
        return opponents_in_lane < 2
    
    def should_i_pass(self):
        """Decide if I should pass instead of shooting"""
        clear_targets = self.find_clear_passing_targets()
        
        if len(clear_targets) > 0 and not self.should_i_shoot():
            return True
        
        if self.is_opponent_nearby(self.ball_2d, radius=1.5):
            return len(clear_targets) > 0
        
        return False
    
    def get_best_pass_target(self):
        """Get the best teammate to pass to"""
        clear_targets = self.find_clear_passing_targets()
        
        if clear_targets:
            best_unum, best_pos, best_score = clear_targets[0]
            return best_pos, best_unum
        else:
            return (15, 0), None


    # ============================================
    # ORIGINAL METHODS
    # ============================================
    
    def GenerateTeamToTargetDistanceArray(self, target, world):
        for teammate in world.teammates:
            pass
    
    def IsFormationReady(self, point_preferences):
        """Check if team is in formation"""
        is_formation_ready = True
        for i in range(1, 6):
            if i != self.active_player_unum: 
                teammate_pos = self.teammate_positions[i-1]

                if not teammate_pos is None:
                    distance = np.sum((teammate_pos - point_preferences[i]) **2)
                    if(distance > 0.3):
                        is_formation_ready = False

        return is_formation_ready

    def GetDirectionRelativeToMyPositionAndTarget(self, target):
        """Get direction to target"""
        target_vec = target - self.my_head_pos_2d
        target_dir = M.vector_angle(target_vec)
        return target_dir
    
   

    def am_i_second_closest_to_ball(self):
        """Check if I'm second closest player"""
        sorted_distances = sorted(enumerate(self.teammates_ball_sq_dist), 
                                key=lambda x: x[1])
        if len(sorted_distances) >= 2:
            second_closest_index = sorted_distances[1][0]
            return (second_closest_index + 1) == self.player_unum
        return False

    def is_ball_in_opponent_half(self):
        """Check if ball is in attacking half"""
        return self.ball_2d[0] > 0
    
    def assign_dynamic_roles(self):
        """
        Decide roles: striker, supporter, defender, etc.
        Roles depend on ball position and player proximity.
        """
        roles = {}
        if self.am_i_closest_to_ball():
            roles[self.player_unum] = "striker"
        elif self.am_i_second_closest_to_ball():
            roles[self.player_unum] = "support"
        elif self.is_ball_in_my_half():
            roles[self.player_unum] = "defense"
        else:
            roles[self.player_unum] = "attack"
        return roles[self.player_unum]
