import numpy as np
import math
from math_ops.Math_Ops import Math_Ops as M
import heapq
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
            
            # Skip if too close
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
                score += forward_progress * 20
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
    
    def astarcalculate_tiki_taka_position(self, base_formation_list, my_unum):
        """
        A* SEARCH for optimal off-ball positioning
        
        Finds best position considering:
        1. Pass availability (2-7m from ball carrier)
        2. Forward progress (closer to goal)
        3. Space from opponents (not crowded)
        4. Spacing from teammates (avoid clustering)
        5. Good passing angles
        
        Args:
            base_formation_list: List of base formation positions
            my_unum: My player number (1-5)
        
        Returns:
            tuple: (x, y) optimal position
        """
        
        # ========================================
        # GOALKEEPER: No A*, fixed positioning
        # ========================================
        if my_unum == 1:
            ball_x = self.ball_2d[0]
            if ball_x < -5:
                gk_x = -13.0
            elif ball_x < 0:
                gk_x = -11.0
            else:
                gk_x = -9.0
            gk_y = np.clip(self.ball_2d[1] * 0.3, -2.5, 2.5)
            return (gk_x, gk_y)
        
        # ========================================
        # BALL CARRIER: Stay just behind ball
        # ========================================
        ball = np.array(self.ball_2d)
        goal = np.array((15, 0))
        ball_carrier_unum = self.active_player_unum
        am_i_carrier = (my_unum == ball_carrier_unum)
        
        if am_i_carrier:
            # Ball carrier doesn't need A*, just position to pass/shoot
            direction_to_goal = goal - ball
            direction_to_goal /= (np.linalg.norm(direction_to_goal) + 1e-5)
            carrier_pos = ball - direction_to_goal * 0.5
            carrier_pos[0] = np.clip(carrier_pos[0], -14.5, 14.5)
            carrier_pos[1] = np.clip(carrier_pos[1], -9.5, 9.5)
            return tuple(carrier_pos)
        
        # ========================================
        # OFF-BALL: Use A* to find optimal position
        # ========================================
        
        my_base_pos = np.array(base_formation_list[my_unum - 1])
        carrier_pos = np.array(self.teammate_positions[ball_carrier_unum - 1]) if ball_carrier_unum else ball
        
        # Grid parameters
        GRID_SIZE = 1.0  # 1 meter resolution
        SEARCH_RADIUS = 8.0  # Search within 8m of base position
        
        # Define search space around base position
        min_x = max(-14.5, my_base_pos[0] - SEARCH_RADIUS)
        max_x = min(14.5, my_base_pos[0] + SEARCH_RADIUS)
        min_y = max(-9.5, my_base_pos[1] - SEARCH_RADIUS)
        max_y = min(9.5, my_base_pos[1] + SEARCH_RADIUS)
        
        def world_to_grid(pos):
            """Convert world coordinates to grid indices"""
            gx = int((pos[0] - min_x) / GRID_SIZE)
            gy = int((pos[1] - min_y) / GRID_SIZE)
            return (gx, gy)
        
        def grid_to_world(grid_pos):
            """Convert grid to world coordinates"""
            wx = grid_pos[0] * GRID_SIZE + min_x
            wy = grid_pos[1] * GRID_SIZE + min_y
            return np.array([wx, wy])
        
        def calculate_position_score(world_pos):
            """
            Score a position (LOWER = BETTER for A*)
            
            Factors:
            1. Distance from carrier (ideal 3-7m)
            2. Distance to goal (closer = better)
            3. Opponent proximity (space = better)
            4. Teammate spacing (avoid clustering)
            5. Passing angle quality
            """
            score = 0
            
            # Factor 1: Distance from ball carrier
            dist_to_carrier = np.linalg.norm(world_pos - carrier_pos)
            
            if dist_to_carrier < 2.0:
                score += 50  # Too close
            elif 3.0 <= dist_to_carrier <= 7.0:
                score += (dist_to_carrier - 5.0) ** 2  # Slight preference for ~5m
            else:
                score += (dist_to_carrier - 7.0) * 5  # Penalty for being too far
            
            # Factor 2: Distance to goal (forward progress)
            dist_to_goal = np.linalg.norm(world_pos - goal)
            score += dist_to_goal * 2.0
            
            # Factor 3: Opponent proximity
            min_opp_dist = float('inf')
            for opp_pos in self.valid_opponent_positions:
                if opp_pos is not None:
                    opp_dist = self.distance(world_pos, opp_pos)
                    min_opp_dist = min(min_opp_dist, opp_dist)
            
            if min_opp_dist < 1.5:
                score += 40  # Very crowded
            elif min_opp_dist < 2.5:
                score += 15  # Somewhat crowded
            # Good spacing: no penalty
            
            # Factor 4: Teammate spacing (avoid clustering)
            for teammate_pos in self.valid_teammate_positions:
                if teammate_pos is not None:
                    teammate_dist = self.distance(world_pos, teammate_pos)
                    if teammate_dist < 2.0:
                        score += 25  # Too close to teammate
                    elif teammate_dist < 3.0:
                        score += 10  # Slightly close
            
            # Factor 5: Passing angle quality
            carrier_to_me = world_pos - carrier_pos
            carrier_to_goal = goal - carrier_pos
            
            if np.linalg.norm(carrier_to_me) > 0.1 and np.linalg.norm(carrier_to_goal) > 0.1:
                carrier_to_me_norm = carrier_to_me / np.linalg.norm(carrier_to_me)
                carrier_to_goal_norm = carrier_to_goal / np.linalg.norm(carrier_to_goal)
                
                # Dot product: 1 = same direction, -1 = opposite
                angle_quality = np.dot(carrier_to_me_norm, carrier_to_goal_norm)
                
                if angle_quality > 0:
                    score -= angle_quality * 15  # Reward forward angles
                else:
                    score += abs(angle_quality) * 8  # Penalty for backward
            
            # Factor 6: Width (prefer some lateral spacing)
            lateral_dist = abs(world_pos[1] - carrier_pos[1])
            if 2.0 <= lateral_dist <= 5.0:
                score -= 8  # Good width
            
            return score
        
        # ========================================
        # A* SEARCH
        # ========================================
        
        start_grid = world_to_grid(my_base_pos)
        
        open_set = []
        start_world = grid_to_world(start_grid)
        start_score = calculate_position_score(start_world)
        heapq.heappush(open_set, (start_score, start_grid))
        
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: start_score}
        
        best_pos = start_world
        best_score = start_score
        
        closed_set = set()
        MAX_ITERATIONS = 150  # Limit for performance
        iterations = 0
        
        # 8-directional movement
        directions = [
            (0, 1), (0, -1), (1, 0), (-1, 0),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]
        
        while open_set and iterations < MAX_ITERATIONS:
            iterations += 1
            
            current_f, current_grid = heapq.heappop(open_set)
            current_world = grid_to_world(current_grid)
            
            # Track best position
            current_score = calculate_position_score(current_world)
            if current_score < best_score:
                best_score = current_score
                best_pos = current_world
            
            # Early termination if excellent position found
            if current_score < 5:
                best_pos = current_world
                break
            
            closed_set.add(current_grid)
            
            # Explore neighbors
            for dx, dy in directions:
                neighbor_grid = (current_grid[0] + dx, current_grid[1] + dy)
                
                if neighbor_grid in closed_set:
                    continue
                
                neighbor_world = grid_to_world(neighbor_grid)
                
                # Check bounds
                if not (-14.5 <= neighbor_world[0] <= 14.5 and 
                        -9.5 <= neighbor_world[1] <= 9.5):
                    continue
                
                # Movement cost (diagonal = 1.414, straight = 1.0)
                move_cost = 1.414 if dx != 0 and dy != 0 else 1.0
                tentative_g = g_score[current_grid] + move_cost
                
                if neighbor_grid not in g_score or tentative_g < g_score[neighbor_grid]:
                    came_from[neighbor_grid] = current_grid
                    g_score[neighbor_grid] = tentative_g
                    
                    h_score = calculate_position_score(neighbor_world)
                    f_score[neighbor_grid] = tentative_g + h_score
                    
                    heapq.heappush(open_set, (f_score[neighbor_grid], neighbor_grid))
        
        # ========================================
        # POST-PROCESSING
        # ========================================
        
        # Ensure defensive players stay behind ball
        if my_base_pos[0] < 0:
            best_pos[0] = min(best_pos[0], ball[0] - 1.0)
        
        # Final bounds check
        best_pos[0] = np.clip(best_pos[0], -14.5, 14.5)
        best_pos[1] = np.clip(best_pos[1], -9.5, 9.5)
        
        return tuple(best_pos)


    def astarfind_best_pass_target(self):
        """
        A*-INSPIRED pass selection
        
        Evaluates each teammate with:
        - Pass difficulty (cost)
        - Position value (heuristic)
        
        Returns:
            tuple: (target_pos, score)
        """
        best_target = None
        best_score = float('inf')  # Lower is better
        opponent_goal = (15, 0)
        
        for i, teammate_pos in enumerate(self.teammate_positions):
            if teammate_pos is None or i == self.player_unum - 1:
                continue
            
            # ===== COST (g): Pass Difficulty =====
            pass_dist = self.distance(self.ball_2d, teammate_pos)
            
            # Distance cost
            if pass_dist < 1.5:
                pass_difficulty = 100  # Too close
            elif 2.0 <= pass_dist <= 5.0:
                pass_difficulty = pass_dist * 2
            elif pass_dist <= 8.0:
                pass_difficulty = 10 + (pass_dist - 5) * 5
            else:
                pass_difficulty = 200  # Too far
            
            # Lane blockage cost
            for opp_pos in self.valid_opponent_positions:
                if opp_pos is not None:
                    line_dist = self.point_to_line_segment_distance(
                        opp_pos, self.ball_2d, teammate_pos
                    )
                    if line_dist < 0.5:
                        pass_difficulty += 40
                    elif line_dist < 1.0:
                        pass_difficulty += 20
                    elif line_dist < 1.5:
                        pass_difficulty += 5
            
            # ===== HEURISTIC (h): Position Value =====
            dist_to_goal = self.distance(teammate_pos, opponent_goal)
            position_value = dist_to_goal * 3
            
            # Central positions bonus
            if abs(teammate_pos[1]) < 3.0:
                position_value -= 10
            
            # Forward positions bonus
            if teammate_pos[0] > 8:
                position_value -= 15
            elif teammate_pos[0] > 0:
                position_value -= 5
            
            # ===== A* SCORE: g + h =====
            total_score = pass_difficulty + position_value
            
            if total_score < best_score:
                best_score = total_score
                best_target = teammate_pos
        
        return best_target, best_score