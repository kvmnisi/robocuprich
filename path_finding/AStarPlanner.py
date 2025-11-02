"""
A*-based calculate_tiki_taka_position()

Replaces your current positioning logic with A* search that finds
optimal position based on:
- Pass availability from ball carrier
- Goal threat
- Opponent spacing
- Teammate spacing
"""

import heapq
import numpy as np
import math


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