from agent.Base_Agent import Base_Agent
from math_ops.Math_Ops import Math_Ops as M
import math
import numpy as np

from strategy.Assignment import role_assignment 
from strategy.Strategy import Strategy 

from formation.Formation import GenerateBasicFormation


class Agent(Base_Agent):
    def __init__(self, host:str, agent_port:int, monitor_port:int, unum:int,
                 team_name:str, enable_log, enable_draw, wait_for_server=True, is_fat_proxy=False) -> None:
        
        # define robot type
        robot_type = (0,1,1,1,2,3,3,3,4,4,4)[unum-1]

        # Initialize base agent
        # Args: Server IP, Agent Port, Monitor Port, Uniform No., Robot Type, Team Name, Enable Log, Enable Draw, play mode correction, Wait for Server, Hear Callback
        super().__init__(host, agent_port, monitor_port, unum, robot_type, team_name, enable_log, enable_draw, True, wait_for_server, None)

        self.enable_draw = enable_draw
        self.state = 0  # 0-Normal, 1-Getting up, 2-Kicking
        self.kick_direction = 0
        self.kick_distance = 0
        self.fat_proxy_cmd = "" if is_fat_proxy else None
        self.fat_proxy_walk = np.zeros(3) # filtered walk parameters for fat proxy

        self.init_pos = ([-14,0],[-9,-5],[-9,0],[-9,5],[-5,-5],[-5,0],[-5,5],[-1,-6],[-1,-2.5],[-1,2.5],[-1,6])[unum-1] # initial formation


    def beam(self, avoid_center_circle=False):
        r = self.world.robot
        pos = self.init_pos[:] # copy position list 
        self.state = 0

        # Avoid center circle by moving the player back 
        if avoid_center_circle and np.linalg.norm(self.init_pos) < 2.5:
            pos[0] = -2.3 

        if np.linalg.norm(pos - r.loc_head_position[:2]) > 0.1 or self.behavior.is_ready("Get_Up"):
            self.scom.commit_beam(pos, M.vector_angle((-pos[0],-pos[1]))) # beam to initial position, face coordinate (0,0)
        else:
            if self.fat_proxy_cmd is None: # normal behavior
                self.behavior.execute("Zero_Bent_Knees_Auto_Head")
            else: # fat proxy behavior
                self.fat_proxy_cmd += "(proxy dash 0 0 0)"
                self.fat_proxy_walk = np.zeros(3) # reset fat proxy walk


    def move(self, target_2d=(0,0), orientation=None, is_orientation_absolute=True,
             avoid_obstacles=True, priority_unums=[], is_aggressive=False, timeout=3000):
        '''
        Walk to target position

        Parameters
        ----------
        target_2d : array_like
            2D target in absolute coordinates
        orientation : float
            absolute or relative orientation of torso, in degrees
            set to None to go towards the target (is_orientation_absolute is ignored)
        is_orientation_absolute : bool
            True if orientation is relative to the field, False if relative to the robot's torso
        avoid_obstacles : bool
            True to avoid obstacles using path planning (maybe reduce timeout arg if this function is called multiple times per simulation cycle)
        priority_unums : list
            list of teammates to avoid (since their role is more important)
        is_aggressive : bool
            if True, safety margins are reduced for opponents
        timeout : float
            restrict path planning to a maximum duration (in microseconds)    
        '''
        r = self.world.robot

        if self.fat_proxy_cmd is not None: # fat proxy behavior
            self.fat_proxy_move(target_2d, orientation, is_orientation_absolute) # ignore obstacles
            return

        if avoid_obstacles:
            target_2d, _, distance_to_final_target = self.path_manager.get_path_to_target(
                target_2d, priority_unums=priority_unums, is_aggressive=is_aggressive, timeout=timeout)
        else:
            distance_to_final_target = np.linalg.norm(target_2d - r.loc_head_position[:2])

        self.behavior.execute("Walk", target_2d, True, orientation, is_orientation_absolute, distance_to_final_target) # Args: target, is_target_abs, ori, is_ori_abs, distance





    def kick(self, kick_direction=None, kick_distance=None, abort=False, enable_pass_command=False):
        '''
        Walk to ball and kick

        Parameters
        ----------
        kick_direction : float
            kick direction, in degrees, relative to the field
        kick_distance : float
            kick distance in meters
        abort : bool
            True to abort.
            The method returns True upon successful abortion, which is immediate while the robot is aligning itself. 
            However, if the abortion is requested during the kick, it is delayed until the kick is completed.
        avoid_pass_command : bool
            When False, the pass command will be used when at least one opponent is near the ball
            
        Returns
        -------
        finished : bool
            Returns True if the behavior finished or was successfully aborted.
        '''
        return self.behavior.execute("Dribble",None,None)

        if self.min_opponent_ball_dist < 1.45 and enable_pass_command:
            self.scom.commit_pass_command()

        self.kick_direction = self.kick_direction if kick_direction is None else kick_direction
        self.kick_distance = self.kick_distance if kick_distance is None else kick_distance

        if self.fat_proxy_cmd is None: # normal behavior
            return self.behavior.execute("Basic_Kick", self.kick_direction, abort) # Basic_Kick has no kick distance control
        else: # fat proxy behavior
            return self.fat_proxy_kick()


    def kickTarget(self, strategyData, mypos_2d=(0,0),target_2d=(0,0), abort=False, enable_pass_command=False):
        '''
        Walk to ball and kick

        Parameters
        ----------
        kick_direction : float
            kick direction, in degrees, relative to the field
        kick_distance : float
            kick distance in meters
        abort : bool
            True to abort.
            The method returns True upon successful abortion, which is immediate while the robot is aligning itself. 
            However, if the abortion is requested during the kick, it is delayed until the kick is completed.
        avoid_pass_command : bool
            When False, the pass command will be used when at least one opponent is near the ball
            
        Returns
        -------
        finished : bool
            Returns True if the behavior finished or was successfully aborted.
        '''
        if target_2d is None or mypos_2d is None:
            return
        # Calculate the vector from the current position to the target position
        vector_to_target = np.array(target_2d) - np.array(mypos_2d)
        
        # Calculate the distance (magnitude of the vector)
        kick_distance = np.linalg.norm(vector_to_target)
        
        # Calculate the direction (angle) in radians
        direction_radians = np.arctan2(vector_to_target[1], vector_to_target[0])
        
        # Convert direction to degrees for easier interpretation (optional)
        kick_direction = np.degrees(direction_radians)


        if strategyData.min_opponent_ball_dist < 1.45 and enable_pass_command:
            self.scom.commit_pass_command()

        self.kick_direction = self.kick_direction if kick_direction is None else kick_direction
        self.kick_distance = self.kick_distance if kick_distance is None else kick_distance

        if self.fat_proxy_cmd is None: # normal behavior
            return self.behavior.execute("Basic_Kick", self.kick_direction, abort) # Basic_Kick has no kick distance control
        else: # fat proxy behavior
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
            self.beam(True) # avoid center circle
        elif self.state == 1 or (behavior.is_ready("Get_Up") and self.fat_proxy_cmd is None):
            self.state = 0 if behavior.execute("Get_Up") else 1
        else:
            if strategyData.play_mode != self.world.M_BEFORE_KICKOFF:
                self.select_skill(strategyData)
            else:
                pass


        #--------------------------------------- 3. Broadcast
        self.radio.broadcast()

        #--------------------------------------- 4. Send to server
        if self.fat_proxy_cmd is None: # normal behavior
            self.scom.commit_and_send( strategyData.robot_model.get_command() )
        else: # fat proxy behavior
            self.scom.commit_and_send( self.fat_proxy_cmd.encode() ) 
            self.fat_proxy_cmd = ""



        



    def select_skill(self, strategyData):
        drawer = self.world.draw
        
        # ========================================
        # PHASE 1: Handle Special Game Modes
        # ========================================
        if not self.is_play_on_mode(strategyData):
            return self.handle_special_game_modes(strategyData)
        
        # ========================================
        # PHASE 2: Role Assignment & Formation
        # ========================================
        formation_positions = GenerateBasicFormation()
        point_preferences = role_assignment(strategyData.teammate_positions, formation_positions)
        strategyData.my_desired_position = point_preferences[strategyData.player_unum]
        strategyData.my_desired_orientation = strategyData.GetDirectionRelativeToMyPositionAndTarget(
            strategyData.my_desired_position
        )
        
        # Visualize formation target
        drawer.line(strategyData.mypos, strategyData.my_desired_position, 2, drawer.Color.blue, "target_line")
        
        # If formation not ready, move to position
        if not strategyData.IsFormationReady(point_preferences):
            drawer.annotation((0, 10.5), "Moving to Formation", drawer.Color.yellow, "status")
            return self.move(strategyData.my_desired_position, orientation=strategyData.my_desired_orientation)
        
        # ========================================
        # PHASE 3: Play Soccer!
        # ========================================
        return self.play_soccer(strategyData)


    def is_play_on_mode(self, strategyData):
        """Check if we're in regular play mode"""
        return strategyData.play_mode == self.world.M_PLAY_ON


    def handle_special_game_modes(self, strategyData):
        """Handle kickoffs, kick-ins, etc."""
        drawer = self.world.draw
        
        # Group game modes
        kickoff_modes = [self.world.M_KICKOFF_LEFT, self.world.M_KICKOFF_RIGHT]
        kickin_modes = [self.world.M_KICK_IN_LEFT, self.world.M_KICK_IN_RIGHT]
        corner_modes = [self.world.M_CORNER_KICK_LEFT, self.world.M_CORNER_KICK_RIGHT]
        goalkick_modes = [self.world.M_GOAL_KICK_LEFT, self.world.M_GOAL_KICK_RIGHT]
        
        drawer.annotation((0, 10.5), f"Mode: {strategyData.play_mode}", drawer.Color.orange, "status")
        
        # For now, simple behavior: closest player goes for ball
        if strategyData.play_mode in kickoff_modes:
            return self.handle_kickoff(strategyData)
        elif strategyData.play_mode in kickin_modes:
            return self.handle_kickin(strategyData)
        elif strategyData.play_mode in corner_modes:
            return self.handle_corner(strategyData)
        elif strategyData.play_mode in goalkick_modes:
            return self.handle_goalkick(strategyData)
        else:
            # Default: hold position
            return self.move(strategyData.my_desired_position)


    def handle_kickoff(self, strategyData):
        """Handle kickoff behavior"""
        # Simple: Player 1 kicks off, others stay back
        if strategyData.player_unum == 1:
            target = (15, 0)  # Kick toward goal
            return self.kickTarget(strategyData, strategyData.mypos, target)
        else:
            # Stay in own half
            return self.move(strategyData.my_desired_position)


    def handle_kickin(self, strategyData):
        """Handle kick-in behavior"""
        # Closest player takes it
        if strategyData.am_i_closest():
            # Find best pass target
            target = self.find_best_pass_target(strategyData)
            return self.kickTarget(strategyData, strategyData.mypos, target)
        else:
            return self.move(strategyData.my_desired_position)


    def handle_corner(self, strategyData):
        """Handle corner kick"""
        # Similar to kick-in for now
        return self.handle_kickin(strategyData)


    def handle_goalkick(self, strategyData):
        """Handle goal kick"""
        # Similar to kick-in for now
        return self.handle_kickin(strategyData)


    def play_soccer(self, strategyData):
        """Main gameplay logic during PlayOn mode"""
        drawer = self.world.draw
        
        # Decision: Am I the one who should go for the ball?
        if strategyData.am_i_closest():
            drawer.annotation(strategyData.mypos, "ATTACKER", drawer.Color.red, f"role_{strategyData.player_unum}")
            return self.be_attacker(strategyData)
        else:
            drawer.annotation(strategyData.mypos, "SUPPORT", drawer.Color.blue, f"role_{strategyData.player_unum}")
            return self.be_supporter(strategyData)


    def be_attacker(self, strategyData):
        """Behavior when I'm going for the ball"""
        drawer = self.world.draw
        
        # Can I kick right now?
        if strategyData.can_i_kick():
            drawer.annotation((0, 10.5), "KICKING", drawer.Color.yellow, "status")
            
            # Decide: Pass or Shoot?
            target = self.decide_kick_target(strategyData)
            
            # Visualize kick target
            drawer.line(strategyData.mypos, target, 3, drawer.Color.red, "kick_line")
            
            return self.kickTarget(strategyData, strategyData.mypos, target)
        else:
            # Move toward ball
            drawer.annotation((0, 10.5), "CHASING BALL", drawer.Color.yellow, "status")
            return self.move(strategyData.ball_abs_pos[:2])


    def be_supporter(self, strategyData):
        """Behavior when I'm NOT going for the ball"""
        drawer = self.world.draw
        
        # Hold formation position, but face the ball
        drawer.clear(f"role_{strategyData.player_unum}")
        drawer.clear("kick_line")
        
        return self.move(
            strategyData.my_desired_position, 
            orientation=strategyData.ball_dir
        )


    def decide_kick_target(self, strategyData):
        """Decide whether to pass or shoot"""
        opponent_goal = (15, 0)
        
        # Simple decision: If I'm close to goal, shoot. Otherwise, pass.
        distance_to_goal = strategyData.distance(strategyData.mypos, opponent_goal)
        
        if distance_to_goal < 8:  # Within shooting range
            return opponent_goal
        else:
            # Find best teammate to pass to
            return self.find_best_pass_target(strategyData)


    def find_best_pass_target(self, strategyData):
        """Find the best teammate to pass to"""
        best_target = (15, 0)  # Default: shoot at goal
        best_score = -999
        
        for i, teammate_pos in enumerate(strategyData.teammate_positions):
            # Don't pass to myself
            if i == strategyData.player_unum - 1:
                continue
            
            # Score this pass option
            score = self.evaluate_pass(strategyData, teammate_pos)
            
            if score > best_score:
                best_score = score
                best_target = teammate_pos
        
        return best_target


    def evaluate_pass(self, strategyData, target_pos):
        """Score a potential pass target (higher is better)"""
        
        # Factor 1: Distance to opponent goal (prefer forward passes)
        opponent_goal = (15, 0)
        target_dist_to_goal = strategyData.distance(target_pos, opponent_goal)
        forward_score = 30 - target_dist_to_goal  # Higher score for positions closer to goal
        
        # Factor 2: Distance to me (prefer not too far)
        pass_distance = strategyData.distance(strategyData.mypos, target_pos)
        distance_score = 10 - abs(pass_distance - 5)  # Prefer ~5m passes
        
        # Factor 3: Is the passing lane clear? (TODO: implement line intersection check)
        # For now, assume clear
        clear_lane_score = 5
        
        total_score = forward_score + distance_score + clear_lane_score
        return total_score
































    

    #--------------------------------------- Fat proxy auxiliary methods


    def fat_proxy_kick(self):
        w = self.world
        r = self.world.robot 
        ball_2d = w.ball_abs_pos[:2]
        my_head_pos_2d = r.loc_head_position[:2]

        if np.linalg.norm(ball_2d - my_head_pos_2d) < 0.25:
            # fat proxy kick arguments: power [0,10]; relative horizontal angle [-180,180]; vertical angle [0,70]
            self.fat_proxy_cmd += f"(proxy kick 10 {M.normalize_deg( self.kick_direction  - r.imu_torso_orientation ):.2f} 20)" 
            self.fat_proxy_walk = np.zeros(3) # reset fat proxy walk
            return True
        else:
            self.fat_proxy_move(ball_2d-(-0.1,0), None, True) # ignore obstacles
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