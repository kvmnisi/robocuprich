from agent.Base_Agent import Base_Agent
from math_ops.Math_Ops import Math_Ops as M
import math
import numpy as np

from strategy.Assignment import role_assignment 
from strategy.Strategy import Strategy 

from formation.Formation import GenerateBasicFormation


from path_finding.AStarPlanner import AStarPlanner


class Agent(Base_Agent):
    def __init__(self, host:str, agent_port:int, monitor_port:int, unum:int,
                 team_name:str, enable_log, enable_draw, wait_for_server=True, is_fat_proxy=False) -> None:
        
        # define robot type
        robot_type = (0,1,1,1,2,3,3,3,4,4,4)[unum-1]

        # Initialize base agent
        super().__init__(host, agent_port, monitor_port, unum, robot_type, team_name, enable_log, enable_draw, True, wait_for_server, None)

        self.enable_draw = enable_draw
        self.state = 0  # 0-Normal, 1-Getting up, 2-Kicking
        self.kick_direction = 0
        self.kick_distance = 0
        self.fat_proxy_cmd = "" if is_fat_proxy else None
        self.fat_proxy_walk = np.zeros(3)

        self.init_pos = ([-14,0],[-9,-5],[-9,0],[-9,5],[-5,-5],[-5,0],[-5,5],[-1,-6],[-1,-2.5],[-1,2.5],[-1,6])[unum-1]

        # NEW: Initialize A* planner
        self.astar_planner = AStarPlanner(
            field_width=30,
            field_height=20,
            grid_resolution=0.5
        )


    def beam(self, avoid_center_circle=False):
        r = self.world.robot
        pos = self.init_pos[:]
        self.state = 0

        if avoid_center_circle and np.linalg.norm(self.init_pos) < 2.5:
            pos[0] = -2.3 

        if np.linalg.norm(pos - r.loc_head_position[:2]) > 0.1 or self.behavior.is_ready("Get_Up"):
            self.scom.commit_beam(pos, M.vector_angle((-pos[0],-pos[1])))
        else:
            if self.fat_proxy_cmd is None:
                self.behavior.execute("Zero_Bent_Knees_Auto_Head")
            else:
                self.fat_proxy_cmd += "(proxy dash 0 0 0)"
                self.fat_proxy_walk = np.zeros(3)


    def move(self, target_2d=(0,0), orientation=None, is_orientation_absolute=True,
             avoid_obstacles=True, priority_unums=[], is_aggressive=False, timeout=3000):
        '''Walk to target position'''
        r = self.world.robot

        if self.fat_proxy_cmd is not None:
            self.fat_proxy_move(target_2d, orientation, is_orientation_absolute)
            return

        if avoid_obstacles:
            target_2d, _, distance_to_final_target = self.path_manager.get_path_to_target(
                target_2d, priority_unums=priority_unums, is_aggressive=is_aggressive, timeout=timeout)
        else:
            distance_to_final_target = np.linalg.norm(target_2d - r.loc_head_position[:2])

        self.behavior.execute("Walk", target_2d, True, orientation, is_orientation_absolute, distance_to_final_target)


    

    def kick(self, kick_direction=None, kick_distance=None, abort=False, enable_pass_command=False):
        '''Walk to ball and kick'''
       # return self.behavior.execute("Dribble",None,None)

        if self.min_opponent_ball_dist < 1.45 and enable_pass_command:
            self.scom.commit_pass_command()

        self.kick_direction = self.kick_direction if kick_direction is None else kick_direction
        self.kick_distance = self.kick_distance if kick_distance is None else kick_distance

        if self.fat_proxy_cmd is None:
            return self.behavior.execute("Basic_Kick", self.kick_direction, abort)
        else:
            return self.fat_proxy_kick()


    def kickTarget(self, strategyData, mypos_2d=(0,0),target_2d=(0,0), abort=False, enable_pass_command=False):
        '''Walk to ball and kick at target'''
        vector_to_target = np.array(target_2d) - np.array(mypos_2d)
        kick_distance = np.linalg.norm(vector_to_target)
        direction_radians = np.arctan2(vector_to_target[1], vector_to_target[0])
        kick_direction = np.degrees(direction_radians)

        if strategyData.min_opponent_ball_dist < 1.45 and enable_pass_command:
            self.scom.commit_pass_command()

        self.kick_direction = self.kick_direction if kick_direction is None else kick_direction
        self.kick_distance = self.kick_distance if kick_distance is None else kick_distance

        if self.fat_proxy_cmd is None:
            return self.behavior.execute("Basic_Kick", self.kick_direction, abort)
        else:
            return self.fat_proxy_kick()


    def think_and_send(self):
        behavior = self.behavior
        strategyData = Strategy(self.world)
        d = self.world.draw

        if strategyData.play_mode == self.world.M_GAME_OVER:
            pass
        elif strategyData.PM_GROUP == self.world.MG_ACTIVE_BEAM:
            self.beam()
        elif strategyData.PM_GROUP == self.world.MG_PASSIVE_BEAM:
            self.beam(True)
        elif self.state == 1 or (behavior.is_ready("Get_Up") and self.fat_proxy_cmd is None):
            self.state = 0 if behavior.execute("Get_Up") else 1
        else:
            if strategyData.play_mode != self.world.M_BEFORE_KICKOFF:
                self.select_skill(strategyData)
            else:
                pass

        self.radio.broadcast()

        if self.fat_proxy_cmd is None:
            self.scom.commit_and_send( strategyData.robot_model.get_command() )
        else:
            self.scom.commit_and_send( self.fat_proxy_cmd.encode() ) 
            self.fat_proxy_cmd = ""


    # def select_skill(self, strategyData):
    #     """
    #     Tiki-Taka main decision function
        
    #     Priority order:
    #     1. Get the ball (highest priority!)
    #     2. Maintain compact shape around ball
    #     3. Create passing opportunities
    #     """
    #     drawer = self.world.draw
        
    #     # ========================================
    #     # PHASE 0: Handle Special Game Modes
    #     # ========================================
    #     if not self.is_play_on_mode(strategyData):
    #         return self.handle_special_game_modes(strategyData)
        
    #     # ========================================
    #     # PHASE 1: BALL IS PRIORITY #1
    #     # ========================================
    #     # Always check ball possession first!
        
    #     if strategyData.am_i_closest_to_ball():
    #         # I'M THE BALL CARRIER - Execute tiki-taka!
    #         drawer.annotation((0, 10.5), "⚽ BALL CARRIER", drawer.Color.red, "status")
    #         return self.execute_tiki_taka_possession(strategyData)
        
    #     # ========================================
    #     # PHASE 2: Support Ball Carrier (Dynamic Formation)
    #     # ========================================
    #     # Not on ball, so position in dynamic formation
        
    #     # Get base formation shape
    #     base_formation = GenerateBasicFormation()
        
    #     # Calculate position that MOVES WITH THE BALL
    #     my_dynamic_position = strategyData.calculate_tiki_taka_position(
    #         base_formation, 
    #         strategyData.player_unum
    #     )
        
    #     # Visualize dynamic formation
    #     drawer.circle(my_dynamic_position, 0.3, 2, drawer.Color.blue, False, 
    #                 f"formation_{strategyData.player_unum}")
    #     drawer.line(strategyData.mypos, my_dynamic_position, 1, drawer.Color.blue,
    #             f"formation_line_{strategyData.player_unum}")
        
    #     # Show my role
    #     ball_dist = strategyData.distance(strategyData.mypos, strategyData.ball_2d)
    #     if ball_dist < 4.0:
    #         role = "CLOSE SUPPORT"
    #         color = drawer.Color.orange
    #     elif ball_dist < 7.0:
    #         role = "MID SUPPORT"
    #         color = drawer.Color.yellow
    #     else:
    #         role = "DEFENSIVE"
    #         color = drawer.Color.blue
        
    #     drawer.annotation(strategyData.mypos, role, color, 
    #                     f"role_{strategyData.player_unum}")
        
    #     # Move to dynamic position, facing ball
    #     return self.move(
    #         target_2d=my_dynamic_position,
    #         orientation=strategyData.ball_dir,
    #         is_orientation_absolute=True,
    #         avoid_obstacles=True,
    #         is_aggressive=False
    #     )

    def select_skill(self, strategyData):
        """
        SIMPLIFIED TIKI-TAKA - Actually Works!
        
        Rules:
        1. Closest player IMMEDIATELY goes for ball
        2. When close enough, pass/shoot QUICKLY
        3. Others hold DYNAMIC formation (moves with ball)
        4. No complex role rotations - just simple, fast decisions
        """
        drawer = self.world.draw
        
        
        if not self.is_play_on_mode(strategyData):
            return self.handle_special_game_modes(strategyData)
        
        if strategyData.am_i_closest_to_ball():
            # YES - I'm attacking the ball!
            drawer.annotation(strategyData.mypos, "⚽ ATTACK", drawer.Color.red, 
                            f"role_{strategyData.player_unum}")
            
            # Can I kick RIGHT NOW?
            if strategyData.can_i_kick():
                # YES - Make QUICK decision and kick!
                drawer.annotation((0, 10.5), "KICKING", drawer.Color.yellow, "status")
                
                # Quick decision: shoot or pass?
                kick_target = self.quick_kick_decision(strategyData)
                
                # Visualize
                distance = strategyData.distance(strategyData.ball_2d, kick_target)
                if kick_target == (15, 0):
                    label = f"SHOOT ({distance:.1f}m)"
                    color = drawer.Color.red
                else:
                    label = f"PASS ({distance:.1f}m)"
                    color = drawer.Color.green
                
                drawer.annotation(strategyData.ball_2d, label, color, "kick_info")
                drawer.line(strategyData.ball_2d, kick_target, 3, color, "kick_line")
                
                # KICK IT!
                return self.kickTarget(strategyData, strategyData.mypos, kick_target)
            
            else:
                # NO - Move to ball
                drawer.annotation((0, 10.5), "CHASING", drawer.Color.orange, "status")
                drawer.clear("kick_info")
                
                return self.move(
                    target_2d=strategyData.ball_2d,
                    orientation=None,
                    avoid_obstacles=True,
                    is_aggressive=True
                )
        
        # ========================================
        # PHASE 2: I'm NOT the ball carrier
        # ========================================
        else:
            # Get base formation
            base_formation = GenerateBasicFormation()
            
            # Calculate MY dynamic position (follows ball)
            my_dynamic_pos = strategyData.calculate_tiki_taka_position(
                base_formation, 
                strategyData.player_unum
            )
            
            # Show role based on distance to ball
            ball_dist = strategyData.distance(strategyData.mypos, strategyData.ball_2d)
            if ball_dist < 4:
                role = "SUPPORT"
                color = drawer.Color.orange
            else:
                role = "DEFEND"
                color = drawer.Color.blue
            
            drawer.annotation(strategyData.mypos, role, color, 
                            f"role_{strategyData.player_unum}")
            
            # Visualize formation position
            drawer.circle(my_dynamic_pos, 0.4, 2, drawer.Color.blue, False,
                        f"formation_{strategyData.player_unum}")
            
            # Move to dynamic position, face ball
            return self.move(
                target_2d=my_dynamic_pos,
                orientation=strategyData.ball_dir,
                avoid_obstacles=True,
                is_aggressive=False
            )


    def quick_kick_decision(self, strategyData):
        """
        FAST kick decision - no complex calculations
        
        Returns:
            tuple: (x, y) position to kick to
        """
        ball_x = strategyData.ball_2d[0]
        opponent_goal = (15, 0)
        
        # Rule 1: If close to goal, SHOOT!
        if ball_x > 8:
            return opponent_goal
        
        # Rule 2: Find closest forward teammate for pass
        best_target = opponent_goal  # Default to shoot
        best_score = -999
        
        for i, teammate_pos in enumerate(strategyData.teammate_positions):
            if teammate_pos is None or i == strategyData.player_unum - 1:
                continue
            
            # Check if forward of ball
            if teammate_pos[0] <= strategyData.ball_2d[0]:
                continue  # Behind ball, skip
            
            # Check distance
            pass_dist = strategyData.distance(strategyData.ball_2d, teammate_pos)
            if pass_dist > 8:
                continue  # Too far
            
            # Simple check: is lane blocked?
            if strategyData.is_passing_lane_blocked(strategyData.ball_2d, teammate_pos, 
                                                    safety_radius=0.6):
                continue  # Blocked
            
            # Score: prefer forward + closer to goal
            score = teammate_pos[0] * 10  # Forward position
            score += (15 - strategyData.distance(teammate_pos, opponent_goal)) * 5
            
            if score > best_score:
                best_score = score
                best_target = teammate_pos
        
        return best_target


    # ========================================
    # GAME MODE HANDLERS - SIMPLIFIED
    # ========================================

    def is_play_on_mode(self, strategyData):
        """Check if in play on mode"""
        return strategyData.play_mode == self.world.M_PLAY_ON


    def handle_special_game_modes(self, strategyData):
        """Handle all special game modes simply"""
        drawer = self.world.draw
        
        # Show mode
        drawer.annotation((0, 10.5), "SET PIECE", drawer.Color.cyan, "status")
        
        # Kickoff modes
        if strategyData.play_mode in [self.world.M_KICKOFF_LEFT, self.world.M_KICKOFF_RIGHT]:
            if strategyData.am_i_closest_to_ball():
                if strategyData.can_i_kick():
                    # Kick to nearest forward teammate
                    return self.kickTarget(strategyData, strategyData.mypos, (5, 0))
                else:
                    return self.move(strategyData.ball_2d)
            else:
                # Hold position
                formation = GenerateBasicFormation()
                my_pos = formation[strategyData.player_unum]
                return self.move(my_pos)
        
        # Kick-in modes
        elif strategyData.play_mode in [self.world.M_KICK_IN_LEFT, self.world.M_KICK_IN_RIGHT]:
            if strategyData.am_i_closest_to_ball():
                if strategyData.can_i_kick():
                    # Quick pass
                    target = self.quick_kick_decision(strategyData)
                    return self.kickTarget(strategyData, strategyData.mypos, target)
                else:
                    return self.move(strategyData.ball_2d)
            else:
                # Get in position for receive
                formation = GenerateBasicFormation()
                my_pos = strategyData.calculate_tiki_taka_position(
                    formation, strategyData.player_unum
                )
                return self.move(my_pos)
        
        # Corner kicks
        elif strategyData.play_mode in [self.world.M_CORNER_KICK_LEFT, self.world.M_CORNER_KICK_RIGHT]:
            if strategyData.am_i_closest_to_ball():
                if strategyData.can_i_kick():
                    return self.kickTarget(strategyData, strategyData.mypos, (12, 0))
                else:
                    return self.move(strategyData.ball_2d)
            else:
                formation = GenerateBasicFormation()
                my_pos = strategyData.calculate_tiki_taka_position(
                    formation, strategyData.player_unum
                )
                return self.move(my_pos)
        
        # Goal kicks
        elif strategyData.play_mode in [self.world.M_GOAL_KICK_LEFT, self.world.M_GOAL_KICK_RIGHT]:
            if strategyData.player_unum == 1:  # Goalkeeper
                if strategyData.can_i_kick():
                    return self.kickTarget(strategyData, strategyData.mypos, (0, 0))
                else:
                    return self.move(strategyData.ball_2d)
            else:
                formation = GenerateBasicFormation()
                my_pos = strategyData.calculate_tiki_taka_position(
                    formation, strategyData.player_unum
                )
                return self.move(my_pos)
        
        # Default: hold formation
        else:
            formation = GenerateBasicFormation()
            my_pos = formation.get(strategyData.player_unum, strategyData.mypos)
            return self.move(my_pos)


    # ========================================
    # GAME MODE HANDLERS (Tiki-Taka Style)
    # ========================================

    def is_play_on_mode(self, strategyData):
        """Check if in regular play"""
        return strategyData.play_mode == self.world.M_PLAY_ON


    def handle_special_game_modes(self, strategyData):
        """Handle set pieces with tiki-taka mentality"""
        drawer = self.world.draw
        
        kickoff_modes = [self.world.M_KICKOFF_LEFT, self.world.M_KICKOFF_RIGHT]
        kickin_modes = [self.world.M_KICK_IN_LEFT, self.world.M_KICK_IN_RIGHT]
        corner_modes = [self.world.M_CORNER_KICK_LEFT, self.world.M_CORNER_KICK_RIGHT]
        goalkick_modes = [self.world.M_GOAL_KICK_LEFT, self.world.M_GOAL_KICK_RIGHT]
        
        drawer.annotation((0, 10.5), f"SET PIECE", drawer.Color.cyan, "status")
        
        if strategyData.play_mode in kickoff_modes:
            return self.handle_kickoff_tiki_taka(strategyData)
        elif strategyData.play_mode in kickin_modes:
            return self.handle_kickin_tiki_taka(strategyData)
        elif strategyData.play_mode in corner_modes:
            return self.handle_corner_tiki_taka(strategyData)
        elif strategyData.play_mode in goalkick_modes:
            return self.handle_goalkick_tiki_taka(strategyData)
        else:
            # Default: Get in shape
            base_formation = GenerateBasicFormation()
            my_pos = strategyData.calculate_tiki_taka_position(
                base_formation, strategyData.player_unum
            )
            return self.move(my_pos)


    def handle_kickoff_tiki_taka(self, strategyData):
        """Kickoff with tiki-taka approach"""
        if strategyData.am_i_closest_to_ball():
            if strategyData.can_i_kick():
                # Short pass to teammate (tiki-taka!)
                targets = strategyData.find_tiki_taka_pass_targets()
                if targets:
                    _, pass_pos, _, _ = targets[0]
                    self.world.draw.annotation(pass_pos, "KICKOFF PASS", 
                                            self.world.draw.Color.green, "kickoff")
                    return self.kickTarget(strategyData, strategyData.mypos, pass_pos)
                else:
                    # No good pass, kick forward
                    return self.kickTarget(strategyData, strategyData.mypos, (5, 0))
            else:
                return self.move(strategyData.ball_2d)
        else:
            # Position for receive
            base_formation = GenerateBasicFormation()
            my_pos = strategyData.calculate_tiki_taka_position(
                base_formation, strategyData.player_unum
            )
            return self.move(my_pos)


    def handle_kickin_tiki_taka(self, strategyData):
        """Kick-in with quick short pass"""
        if strategyData.am_i_closest_to_ball():
            if strategyData.can_i_kick():
                # Find SHORT pass option (tiki-taka!)
                targets = strategyData.find_tiki_taka_pass_targets()
                if targets:
                    _, pass_pos, _, pass_type = targets[0]
                    label = f"KICK-IN: {pass_type}"
                    self.world.draw.annotation(pass_pos, label, 
                                            self.world.draw.Color.green, "kickin")
                    return self.kickTarget(strategyData, strategyData.mypos, pass_pos)
                else:
                    # Safe clear
                    return self.kickTarget(strategyData, strategyData.mypos, 
                                        (strategyData.ball_2d[0] + 3, 0))
            else:
                return self.move(strategyData.ball_2d)
        else:
            # Position for receive
            base_formation = GenerateBasicFormation()
            my_pos = strategyData.calculate_tiki_taka_position(
                base_formation, strategyData.player_unum
            )
            return self.move(my_pos)


    def handle_corner_tiki_taka(self, strategyData):
        """Corner kick"""
        return self.handle_kickin_tiki_taka(strategyData)


    def handle_goalkick_tiki_taka(self, strategyData):
        """Goal kick"""
        return self.handle_kickin_tiki_taka(strategyData)

    def handle_all_play_modes(self, strategyData):
        """
        Basic handlers for all play modes
        """
        drawer = self.world.draw
        mypos = strategyData.mypos
        ball = strategyData.ball_2d
        
        # Show current play mode
        mode_name = self.get_play_mode_name(strategyData.play_mode)
        drawer.annotation((0, 11), f"MODE: {mode_name}", drawer.Color.cyan, "play_mode")
        
        # BEFORE KICKOFF - Position in formation
        if strategyData.play_mode == self.world.M_BEFORE_KICKOFF:
            formation = GenerateBasicFormation()
            target_pos = formation[strategyData.player_unum - 1]
            return self.move(target_2d=target_pos, orientation=0)
        
        # KICKOFF (OUR SIDE)
        elif strategyData.play_mode in [self.world.M_KICKOFF_LEFT, self.world.M_KICKOFF_RIGHT]:
            if strategyData.am_i_closest_to_ball():
                if strategyData.can_i_kick():
                    # Kick forward to start play
                    return self.kickTarget(strategyData, mypos, (5, 0))
                else:
                    return self.move(target_2d=ball, orientation=strategyData.ball_dir)
            else:
                # Spread out for kickoff
                formation = GenerateBasicFormation()
                target_pos = strategyData.calculate_tiki_taka_position(formation, strategyData.player_unum)
                return self.move(target_2d=target_pos, orientation=strategyData.ball_dir)
        
        # KICKOFF (THEIR SIDE) - Get in defensive positions
        elif strategyData.play_mode in [self.world.M_KICKOFF_RIGHT, self.world.M_KICKOFF_LEFT]:
            formation = GenerateBasicFormation()
            target_pos = strategyData.calculate_tiki_taka_position(formation, strategyData.player_unum)
            # Stay more defensive during opponent kickoff
            target_pos = (target_pos[0] - 2.0, target_pos[1])
            return self.move(target_2d=target_pos, orientation=strategyData.ball_dir)
        
        # KICK-IN (OUR SIDE)
        elif strategyData.play_mode in [self.world.M_KICK_IN_LEFT, self.world.M_KICK_IN_RIGHT]:
            if strategyData.am_i_closest_to_ball():
                if strategyData.can_i_kick():
                    # Pass to nearest teammate
                    targets = strategyData.find_tiki_taka_pass_targets()
                    if targets:
                        _, pass_pos, _, _ = targets[0]
                        return self.kickTarget(strategyData, mypos, pass_pos)
                    else:
                        return self.kickTarget(strategyData, mypos, (ball[0] + 3, ball[1]))
                else:
                    return self.move(target_2d=ball, orientation=strategyData.ball_dir)
            else:
                # Get open for pass
                formation = GenerateBasicFormation()
                target_pos = strategyData.calculate_tiki_taka_position(formation, strategyData.player_unum)
                return self.move(target_2d=target_pos, orientation=strategyData.ball_dir)
        
        # KICK-IN (THEIR SIDE) - Mark opponents
        elif strategyData.play_mode in [self.world.M_KICK_IN_RIGHT, self.world.M_KICK_IN_LEFT]:
            formation = GenerateBasicFormation()
            target_pos = strategyData.calculate_tiki_taka_position(formation, strategyData.player_unum)
            return self.move(target_2d=target_pos, orientation=strategyData.ball_dir)
        
        # CORNER KICK (OUR SIDE)
        elif strategyData.play_mode in [self.world.M_CORNER_KICK_LEFT, self.world.M_CORNER_KICK_RIGHT]:
            if strategyData.am_i_closest_to_ball():
                if strategyData.can_i_kick():
                    # Cross to goal area
                    return self.kickTarget(strategyData, mypos, (10, 0))
                else:
                    return self.move(target_2d=ball, orientation=strategyData.ball_dir)
            else:
                # Get in scoring positions
                if strategyData.player_unum in [4, 5]:  # Attackers go to goal area
                    target_pos = (10, 2 if strategyData.player_unum == 4 else -2)
                else:  # Others stay back
                    formation = GenerateBasicFormation()
                    target_pos = strategyData.calculate_tiki_taka_position(formation, strategyData.player_unum)
                return self.move(target_2d=target_pos, orientation=strategyData.ball_dir)
        
        # CORNER KICK (THEIR SIDE) - Defensive positioning
        elif strategyData.play_mode in [self.world.M_CORNER_KICK_RIGHT, self.world.M_CORNER_KICK_LEFT]:
            # Defend near our goal
            if strategyData.player_unum == 1:  # GK stays in goal
                return self.move(target_2d=(-13, 0), orientation=0)
            else:
                # Defenders form defensive line
                defensive_positions = {
                    2: (-10, -3), 3: (-10, 0), 4: (-10, 3), 5: (-8, 0)
                }
                target_pos = defensive_positions.get(strategyData.player_unum, (-9, 0))
                return self.move(target_2d=target_pos, orientation=strategyData.ball_dir)
        
        # GOAL KICK (OUR SIDE)
        elif strategyData.play_mode in [self.world.M_GOAL_KICK_LEFT, self.world.M_GOAL_KICK_RIGHT]:
            if strategyData.player_unum == 1:  # GK takes goal kick
                if strategyData.can_i_kick():
                    # Pass to defender
                    defender_pos = strategyData.teammate_positions[1]  # Player 2
                    if defender_pos is not None:
                        return self.kickTarget(strategyData, mypos, defender_pos)
                    else:
                        return self.kickTarget(strategyData, mypos, (ball[0] + 5, 0))
                else:
                    return self.move(target_2d=ball, orientation=strategyData.ball_dir)
            else:
                # Spread out for pass
                formation = GenerateBasicFormation()
                target_pos = strategyData.calculate_tiki_taka_position(formation, strategyData.player_unum)
                return self.move(target_2d=target_pos, orientation=strategyData.ball_dir)
        
        # GOAL KICK (THEIR SIDE) - Push up
        elif strategyData.play_mode in [self.world.M_GOAL_KICK_RIGHT, self.world.M_GOAL_KICK_LEFT]:
            formation = GenerateBasicFormation()
            target_pos = strategyData.calculate_tiki_taka_position(formation, strategyData.player_unum)
            # Push forward during opponent goal kick
            target_pos = (target_pos[0] + 2.0, target_pos[1])
            return self.move(target_2d=target_pos, orientation=strategyData.ball_dir)
        
        # FREE KICK (OUR SIDE)
        elif strategyData.play_mode in [self.world.M_FREE_KICK_LEFT, self.world.M_FREE_KICK_RIGHT]:
            if strategyData.am_i_closest_to_ball():
                if strategyData.can_i_kick():
                    # Quick free kick
                    targets = strategyData.find_tiki_taka_pass_targets()
                    if targets:
                        _, pass_pos, _, _ = targets[0]
                        return self.kickTarget(strategyData, mypos, pass_pos)
                    else:
                        return self.kickTarget(strategyData, mypos, (ball[0] + 3, ball[1]))
                else:
                    return self.move(target_2d=ball, orientation=strategyData.ball_dir)
            else:
                # Get open for pass
                formation = GenerateBasicFormation()
                target_pos = strategyData.calculate_tiki_taka_position(formation, strategyData.player_unum)
                return self.move(target_2d=target_pos, orientation=strategyData.ball_dir)
        
        # FREE KICK (THEIR SIDE) - Form defensive wall
        elif strategyData.play_mode in [self.world.M_FREE_KICK_RIGHT, self.world.M_FREE_KICK_LEFT]:
            # Basic defensive positioning
            formation = GenerateBasicFormation()
            target_pos = strategyData.calculate_tiki_taka_position(formation, strategyData.player_unum)
            return self.move(target_2d=target_pos, orientation=strategyData.ball_dir)
        
        # OFFSIDE - Wait for restart
        elif strategyData.play_mode in [self.world.M_OFFSIDE_LEFT, self.world.M_OFFSIDE_RIGHT]:
            formation = GenerateBasicFormation()
            target_pos = formation[strategyData.player_unum - 1]
            return self.move(target_2d=target_pos, orientation=0)
        
        # GAME OVER - Stop moving
        elif strategyData.play_mode == self.world.M_GAME_OVER:
            return self.move(target_2d=mypos, orientation=0)
        
        # Default fallback - use formation
        else:
            formation = GenerateBasicFormation()
            target_pos = strategyData.calculate_tiki_taka_position(formation, strategyData.player_unum)
            return self.move(target_2d=target_pos, orientation=strategyData.ball_dir)


    def get_play_mode_name(self, play_mode):
        """Convert play mode constant to readable name"""
        mode_names = {
            self.world.M_BEFORE_KICKOFF: "BEFORE KICKOFF",
            self.world.M_PLAY_ON: "PLAY ON",
            self.world.M_KICKOFF_LEFT: "KICKOFF LEFT",
            self.world.M_KICKOFF_RIGHT: "KICKOFF RIGHT", 
            self.world.M_KICK_IN_LEFT: "KICK-IN LEFT",
            self.world.M_KICK_IN_RIGHT: "KICK-IN RIGHT",
            self.world.M_CORNER_KICK_LEFT: "CORNER LEFT",
            self.world.M_CORNER_KICK_RIGHT: "CORNER RIGHT",
            self.world.M_GOAL_KICK_LEFT: "GOAL KICK LEFT",
            self.world.M_GOAL_KICK_RIGHT: "GOAL KICK RIGHT",
            self.world.M_FREE_KICK_LEFT: "FREE KICK LEFT",
            self.world.M_FREE_KICK_RIGHT: "FREE KICK RIGHT",
            self.world.M_OFFSIDE_LEFT: "OFFSIDE LEFT", 
            self.world.M_OFFSIDE_RIGHT: "OFFSIDE RIGHT",
            self.world.M_GAME_OVER: "GAME OVER"
        }
        return mode_names.get(play_mode, "UNKNOWN MODE")























    # Keep existing fat proxy methods
    def fat_proxy_kick(self):
        w = self.world
        r = self.world.robot 
        ball_2d = w.ball_abs_pos[:2]
        my_head_pos_2d = r.loc_head_position[:2]

        if np.linalg.norm(ball_2d - my_head_pos_2d) < 0.25:
            self.fat_proxy_cmd += f"(proxy kick 10 {M.normalize_deg( self.kick_direction  - r.imu_torso_orientation ):.2f} 20)" 
            self.fat_proxy_walk = np.zeros(3)
            return True
        else:
            self.fat_proxy_move(ball_2d-(-0.1,0), None, True)
            return False


    def fat_proxy_move(self, target_2d, orientation, is_orientation_absolute):
        r = self.world.robot

        target_dist = np.linalg.norm(target_2d - r.loc_head_position[:2])
        target_dir = M.target_rel_angle(r.loc_head_position[:2], r.imu_torso_orientation, target_2d)

        if target_dist > 0.1 and abs(target_dir) < 8:
            self.fat_proxy_cmd += (f"(proxy dash {100} {0} {0})")
            return

        if target_dist < 0.1:
            if is_orientation_absolute:
                orientation = M.normalize_deg( orientation - r.imu_torso_orientation )
            target_dir = np.clip(orientation, -60, 60)
            self.fat_proxy_cmd += (f"(proxy dash {0} {0} {target_dir:.1f})")
        else:
            self.fat_proxy_cmd += (f"(proxy dash {20} {0} {target_dir:.1f})")