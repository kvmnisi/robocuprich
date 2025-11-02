from agent.Base_Agent import Base_Agent
from math_ops.Math_Ops import Math_Ops as M
import math
import numpy as np

from strategy.Assignment import role_assignment 
from strategy.Strategy import Strategy 

from formation.Formation import GenerateBasicFormation
from formation.Formation import KickOffFormation


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

        self.init_pos = ([-14,0],[-6, 0],[-2,5],[-2,-4],[-1,-0])[unum-1]


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
        return self.behavior.execute("Dribble",None,None)

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
            self.select_skill(strategyData)

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
        TIKI-TAKA with ROLE ASSIGNMENT
        
        Flow:
        1. Active player attacks ball
        2. Others use role assignment for positioning
        """
        drawer = self.world.draw
        
        # Handle non-play-on modes
        if not self.is_play_on_mode(strategyData):
            return self.handle_all_play_modes(strategyData)
        
        # ========================================
        # ROLE ASSIGNMENT PHASE
        # ========================================
        # Get base formation
        formation_positions = GenerateBasicFormation(strategyData.ball_2d)
        
        # Calculate dynamic formation (moves with ball)
        dynamic_formation = []
        for unum in range(1, 6):
            dynamic_pos = strategyData.calculate_tiki_taka_position(formation_positions, unum)
            dynamic_formation.append(dynamic_pos)
        
        # Use role assignment to match players to dynamic positions
        
        point_preferences = role_assignment(strategyData.teammate_positions, dynamic_formation)
        
        # Update my desired position
        strategyData.my_desired_position = point_preferences[strategyData.player_unum]
        strategyData.my_desired_orientation = strategyData.GetDirectionRelativeToMyPositionAndTarget(
            strategyData.my_desired_position
        )
        
        # Visualize assignment
        drawer.line(strategyData.mypos, strategyData.my_desired_position, 2, 
                    drawer.Color.blue, "target_line")
        # drawer.circle(strategyData.my_desired_position, 0.4, 2, drawer.Color.blue, False,
        #             f"assigned_pos_{strategyData.player_unum}")
        
        # ========================================
        # PHASE 1: Am I attacking the ball?
        # ========================================
        if strategyData.am_i_closest_to_ball():
            drawer.annotation((0, 10.5), "ATTACKING BALL", drawer.Color.red, "status")
            drawer.annotation(strategyData.mypos, "⚽ ATTACKER", drawer.Color.red, 
                            f"role_{strategyData.player_unum}")
            
            # Can I kick NOW?
            if strategyData.can_i_kick():
                drawer.annotation((0, 10.5), "KICKING", drawer.Color.yellow, "status")
                
                # Make smart kick decision
                kick_target = self.kick_decision(strategyData, drawer)
                
                # KICK!
                return self.kickTarget(strategyData, strategyData.mypos, kick_target)
            
            else:
                # Chase ball
                drawer.annotation((0, 10.5), "CHASING BALL", drawer.Color.orange, "status")
                drawer.clear("kick_info")
                
                return self.move(
                    target_2d=strategyData.ball_2d,
                    orientation=None,
                    avoid_obstacles=True,
                    is_aggressive=True
                )
        
        # ========================================
        # PHASE 2: Support (Use Role Assignment)
        # ========================================
        else:
            drawer.annotation((0, 10.5), "Role Assignment Phase", drawer.Color.yellow, "status")
            
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
            
            # Check if formation is ready (optional - can remove this)
            if not strategyData.IsFormationReady(point_preferences):
                # Still moving to position
                return self.move(
                    target_2d=strategyData.my_desired_position,
                    orientation=strategyData.my_desired_orientation
                )
            
            # Move to assigned position, face ball
            return self.move(
                target_2d=strategyData.my_desired_position,
                orientation=strategyData.ball_dir,  # Face ball for quick reaction
                avoid_obstacles=True,
                is_aggressive=False
            )


    def kick_decision(self, strategyData, drawer):
        """
        Smart kick decision: shoot or pass?
        
        Returns:
            tuple: (x, y) position to kick towards
        """
        opponent_goal = np.array([15.0, 0.0])
        
        # Should we shoot?
        if strategyData.should_shoot():
            drawer.annotation(strategyData.ball_2d, "SHOOT!", drawer.Color.red, "kick_info")
            drawer.line(strategyData.ball_2d, opponent_goal, 3, drawer.Color.red, "kick_line")
            return tuple(opponent_goal)
        
        # Find best pass
        pass_target, pass_score = strategyData.find_best_pass_target()
        
        if pass_target is not None and pass_score > 0:
            # Good pass available
            distance = strategyData.distance(strategyData.ball_2d, pass_target)
            drawer.annotation(strategyData.ball_2d, f"PASS ({distance:.1f}m)", 
                            drawer.Color.green, "kick_info")
            drawer.line(strategyData.ball_2d, pass_target, 3, drawer.Color.green, "kick_line")
            return tuple(pass_target)
        
        # No good options - just kick forward
        forward_target = np.array([strategyData.ball_2d[0] + 5, strategyData.ball_2d[1]])
        drawer.annotation(strategyData.ball_2d, "CLEAR", drawer.Color.yellow, "kick_info")
        drawer.line(strategyData.ball_2d, forward_target, 3, drawer.Color.yellow, "kick_line")
        return tuple(forward_target)


    # ========================================
    # GAME MODE HANDLERS - SIMPLIFIED
    # ========================================

    def is_play_on_mode(self, strategyData):
        """Check if in play on mode"""
        return strategyData.play_mode == self.world.M_PLAY_ON


    

    # ========================================
    # GAME MODE HANDLERS (Tiki-Taka Style)
    # ========================================

    def is_play_on_mode(self, strategyData):
        """Check if in regular play"""
        return strategyData.play_mode == self.world.M_PLAY_ON


    

    def handle_all_play_modes(self, strategyData):
        """
        Basic handlers for all play modes - compatible with World.py constants
        """
        drawer = self.world.draw
        mypos = strategyData.mypos
        ball = strategyData.ball_2d
        
        # Show current play mode
        mode_name = self.get_play_mode_name(strategyData.play_mode)
        drawer.annotation((0, 11), f"MODE: {mode_name}", drawer.Color.cyan, "play_mode")
        
        # BEFORE KICKOFF - Position in formation
        if strategyData.play_mode == self.world.M_BEFORE_KICKOFF:
            formation = KickOffFormation()
            target_pos = formation[strategyData.player_unum - 1]
            return self.move(target_2d=target_pos, orientation=0)
        
        # OUR KICKOFF
        elif strategyData.play_mode == self.world.M_OUR_KICKOFF:
            # Define kickoff formation
            formation  = [
            (-13, 0),     # Player 1: GK
            (-6, 0),      # Player 2: Midfielder (center back)
            (-2, 5),      # Player 3: Left mid
            (-2, -4),     # Player 4: Right mid
            (-1, 0)     # Player 5: Striker (top of diamond)
            ]
            
            # Use role assignment for kickoff positions
            from strategy.Assignment import role_assignment
            point_preferences = role_assignment(strategyData.teammate_positions, formation)
            
            # Update my assigned position
            strategyData.my_desired_position = point_preferences[strategyData.player_unum]
            strategyData.my_desired_orientation = 180 if strategyData.am_i_closest_to_ball() else 0
            
            # Visualize assignment
            drawer.line(strategyData.mypos, strategyData.my_desired_position, 2, 
                        drawer.Color.blue, "kickoff_assignment")
            
            # Check if formation is ready
            if not strategyData.IsFormationReady(point_preferences):
                # Still getting into position
                drawer.annotation((0, 10.5), "KICKOFF SETUP", drawer.Color.yellow, "status")
                
                # Player taking kickoff faces own goal
                orientation = 180 if strategyData.am_i_closest_to_ball() else 0
                
                return self.move(
                    target_2d=strategyData.my_desired_position,
                    orientation=orientation
                )
            
            # Formation is ready - execute kickoff
            if strategyData.am_i_closest_to_ball():
                drawer.annotation((0, 10.5), "KICKOFF!", drawer.Color.green, "status")
                
                if strategyData.can_i_kick():
                    # Find player at position (-2, 0) from point_preferences
                    backward_pass_target = None
                    for unum, pos in point_preferences.items():
                        # Find midfielder behind me (around x=-2)
                        if pos[0] < -1.5 and pos[0] > -3.0 and abs(pos[1]) < 1.0:
                            # Get actual position of that player
                            backward_pass_target = strategyData.teammate_positions[unum - 1]
                            break
                    
                    # Default to fixed position if not found
                    if backward_pass_target is None:
                        backward_pass_target = (-2, 0)
                    
                    drawer.annotation(strategyData.ball_2d, "BACKWARD PASS", drawer.Color.green, "kickoff_pass")
                    drawer.line(strategyData.ball_2d, backward_pass_target, 3, drawer.Color.green, "kickoff_line")
                    
                    return self.kickTarget(strategyData, mypos, backward_pass_target)
                else:
                    # Move to ball, face own goal
                    return self.move(target_2d=ball, orientation=180)
            else:
                # Wait in position for kickoff pass
                drawer.annotation((0, 10.5), "READY FOR KICKOFF", drawer.Color.cyan, "status")
                return self.move(
                    target_2d=strategyData.my_desired_position,
                    orientation=0  # Face forward to receive
                )
        
        # THEIR KICKOFF - Get in defensive positions
        elif strategyData.play_mode == self.world.M_THEIR_KICKOFF:
            formation = KickOffFormation()
            target_pos = strategyData.calculate_tiki_taka_position(formation, strategyData.player_unum)
            # Stay more defensive during opponent kickoff
            target_pos = (target_pos[0] - 2.0, target_pos[1])
            return self.move(target_2d=target_pos, orientation=strategyData.ball_dir)
        
        # OUR KICK-IN
        elif strategyData.play_mode == self.world.M_OUR_KICK_IN:
            if strategyData.am_i_closest_to_ball():
                if strategyData.can_i_kick():
                    # Pass to nearest teammate
                    targets = strategyData.find_best_pass_target()
                    if targets:
                        _, pass_pos, _, _ = targets[0]
                        return self.kickTarget(strategyData, mypos, pass_pos)
                    else:
                        return self.kickTarget(strategyData, mypos, (ball[0] + 3, ball[1]))
                else:
                    return self.move(target_2d=ball, orientation=strategyData.ball_dir)
            else:
                # Get open for pass
                formation = GenerateBasicFormation(strategyData.ball_2d)
                target_pos = strategyData.calculate_tiki_taka_position(formation, strategyData.player_unum)
                return self.move(target_2d=target_pos, orientation=strategyData.ball_dir)
        
        # THEIR KICK-IN - Mark opponents
        elif strategyData.play_mode == self.world.M_THEIR_KICK_IN:
            formation = GenerateBasicFormation(strategyData.ball_2d)
            target_pos = strategyData.calculate_tiki_taka_position(formation, strategyData.player_unum)
            return self.move(target_2d=target_pos, orientation=strategyData.ball_dir)
        
        # OUR CORNER KICK
        elif strategyData.play_mode == self.world.M_OUR_CORNER_KICK:
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
                    formation = GenerateBasicFormation(strategyData.ball_2d)
                    target_pos = strategyData.calculate_tiki_taka_position(formation, strategyData.player_unum)
                return self.move(target_2d=target_pos, orientation=strategyData.ball_dir)
        
        # THEIR CORNER KICK - Defensive positioning
        elif strategyData.play_mode == self.world.M_THEIR_CORNER_KICK:
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
        
        # OUR GOAL KICK
        elif strategyData.play_mode == self.world.M_OUR_GOAL_KICK:
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
                formation = GenerateBasicFormation(strategyData.ball_2d)
                target_pos = strategyData.calculate_tiki_taka_position(formation, strategyData.player_unum)
                return self.move(target_2d=target_pos, orientation=strategyData.ball_dir)
        
        # THEIR GOAL KICK - Push up
        elif strategyData.play_mode == self.world.M_THEIR_GOAL_KICK:
            formation = GenerateBasicFormation(strategyData.ball_2d)
            target_pos = strategyData.calculate_tiki_taka_position(formation, strategyData.player_unum)
            # Push forward during opponent goal kick
            target_pos = (target_pos[0] + 2.0, target_pos[1])
            return self.move(target_2d=target_pos, orientation=strategyData.ball_dir)
        
        # OUR FREE KICK
        elif strategyData.play_mode in [self.world.M_OUR_FREE_KICK, self.world.M_OUR_DIR_FREE_KICK]:
            if strategyData.am_i_closest_to_ball():
                if strategyData.can_i_kick():
                    # Quick free kick
                    pass_target, pass_score = strategyData.find_best_pass_target()
                    if pass_target is not None:
                        
                        return self.kickTarget(strategyData, mypos, pass_target)
                    else:
                        return self.kickTarget(strategyData, mypos, (ball[0] + 3, ball[1]))
                else:
                    return self.move(target_2d=ball, orientation=strategyData.ball_dir)
            else:
                # Get open for pass
                formation = GenerateBasicFormation(strategyData.ball_2d)
                target_pos = strategyData.calculate_tiki_taka_position(formation, strategyData.player_unum)
                return self.move(target_2d=target_pos, orientation=strategyData.ball_dir)
        
        # THEIR FREE KICK - Form defensive wall
        elif strategyData.play_mode in [self.world.M_THEIR_FREE_KICK, self.world.M_THEIR_DIR_FREE_KICK]:
            # Basic defensive positioning
            formation = GenerateBasicFormation(strategyData.ball_2d)
            target_pos = strategyData.calculate_tiki_taka_position(formation, strategyData.player_unum)
            return self.move(target_2d=target_pos, orientation=strategyData.ball_dir)
        
        # OFFSIDE - Wait for restart
        elif strategyData.play_mode in [self.world.M_OUR_OFFSIDE, self.world.M_THEIR_OFFSIDE]:
            formation = GenerateBasicFormation(strategyData.ball_2d)
            target_pos = formation[strategyData.player_unum - 1]
            return self.move(target_2d=target_pos, orientation=0)
        
        # GAME OVER - Stop moving
        elif strategyData.play_mode == self.world.M_GAME_OVER:
            return self.move(target_2d=mypos, orientation=0)
        
        # OUR GOAL / THEIR GOAL - Beam to positions
        elif strategyData.play_mode in [self.world.M_OUR_GOAL, self.world.M_THEIR_GOAL]:
            # Use beam for goal situations (handled in think_and_send)
            formation = GenerateBasicFormation(strategyData.ball_2d)
            target_pos = formation[strategyData.player_unum - 1]
            return self.move(target_2d=target_pos, orientation=0)
        
        # Default fallback - use formation
        else:
            formation = GenerateBasicFormation(strategyData.ball_2d)
            target_pos = strategyData.calculate_tiki_taka_position(formation, strategyData.player_unum)
            return self.move(target_2d=target_pos, orientation=strategyData.ball_dir)


    def get_play_mode_name(self, play_mode):
        """Convert play mode constant to readable name"""
        mode_names = {
            self.world.M_BEFORE_KICKOFF: "BEFORE KICKOFF",
            self.world.M_PLAY_ON: "PLAY ON",
            self.world.M_GAME_OVER: "GAME OVER",
            
            self.world.M_OUR_KICKOFF: "OUR KICKOFF",
            self.world.M_OUR_KICK_IN: "OUR KICK-IN", 
            self.world.M_OUR_CORNER_KICK: "OUR CORNER",
            self.world.M_OUR_GOAL_KICK: "OUR GOAL KICK",
            self.world.M_OUR_FREE_KICK: "OUR FREE KICK",
            self.world.M_OUR_DIR_FREE_KICK: "OUR DIR FREE KICK",
            self.world.M_OUR_PASS: "OUR PASS",
            self.world.M_OUR_GOAL: "OUR GOAL",
            self.world.M_OUR_OFFSIDE: "OUR OFFSIDE",
            
            self.world.M_THEIR_KICKOFF: "THEIR KICKOFF",
            self.world.M_THEIR_KICK_IN: "THEIR KICK-IN",
            self.world.M_THEIR_CORNER_KICK: "THEIR CORNER", 
            self.world.M_THEIR_GOAL_KICK: "THEIR GOAL KICK",
            self.world.M_THEIR_FREE_KICK: "THEIR FREE KICK",
            self.world.M_THEIR_DIR_FREE_KICK: "THEIR DIR FREE KICK",
            self.world.M_THEIR_PASS: "THEIR PASS",
            self.world.M_THEIR_GOAL: "THEIR GOAL",
            self.world.M_THEIR_OFFSIDE: "THEIR OFFSIDE"
        }
        return mode_names.get(play_mode, f"UNKNOWN ({play_mode})")























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