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


    # NEW METHOD: A* movement
    def move_with_astar(self, strategyData, target_2d, orientation=None, 
                        is_orientation_absolute=True, avoid_opponents=True, 
                        avoid_teammates=False, aggressive_mode=False, 
                        visualize=True):
        """Move to target using A* path planning"""
        drawer = self.world.draw
        
        # Collect obstacles
        obstacles = []
        
        if avoid_opponents and strategyData.valid_opponent_positions:
            obstacles.extend(strategyData.valid_opponent_positions)
        
        if avoid_teammates and strategyData.valid_teammate_positions:
            for i, teammate_pos in enumerate(strategyData.valid_teammate_positions):
                if teammate_pos is not None and i != strategyData.player_unum - 1:
                    obstacles.append(teammate_pos)
        
        # Plan path with A*
        path, success = self.astar_planner.plan_path(
            start_pos=strategyData.mypos[:2],
            goal_pos=target_2d,
            obstacles=obstacles,
            aggressive_mode=aggressive_mode,
            max_iterations=500
        )
        
        # Visualize path
        if self.enable_draw and visualize and len(path) > 1:
            for i in range(20):
                drawer.clear(f"astar_path_{self.world.robot.unum}_{i}")
            
            for i in range(len(path) - 1):
                drawer.line(path[i], path[i+1], 2, drawer.Color.green, 
                           f"astar_path_{self.world.robot.unum}_{i}")
            
            drawer.circle(target_2d, 0.3, 2, drawer.Color.yellow, True, 
                         f"astar_goal_{self.world.robot.unum}")
        
        # Move to next waypoint
        if len(path) >= 2:
            next_waypoint = path[1]
        else:
            next_waypoint = target_2d
        
        return self.move(next_waypoint, orientation, is_orientation_absolute, 
                        avoid_obstacles=False)


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


    def select_skill(self, strategyData):
        """Main decision function - called every game cycle"""
        drawer = self.world.draw
        
        # PHASE 1: Handle Special Game Modes
        if not self.is_play_on_mode(strategyData):
            return self.handle_special_game_modes(strategyData)
        
        # PHASE 2: Role Assignment & Formation
        formation_positions = GenerateBasicFormation()
        point_preferences = role_assignment(strategyData.teammate_positions, formation_positions)
        strategyData.my_desired_position = point_preferences[strategyData.player_unum]
        strategyData.my_desired_orientation = strategyData.GetDirectionRelativeToMyPositionAndTarget(
            strategyData.my_desired_position
        )
        
        drawer.line(strategyData.mypos, strategyData.my_desired_position, 2, 
                   drawer.Color.blue, f"target_line_{strategyData.player_unum}")
        
        if not strategyData.IsFormationReady(point_preferences):
            drawer.annotation((0, 10.5), "Moving to Formation", drawer.Color.yellow, "status")
            return self.move_with_astar(
                strategyData,
                target_2d=strategyData.my_desired_position,
                orientation=strategyData.my_desired_orientation,
                avoid_opponents=True,
                avoid_teammates=True,
                aggressive_mode=False
            )
        
        # PHASE 3: Play Soccer
        return self.play_soccer(strategyData)


    def is_play_on_mode(self, strategyData):
        """Check if we're in regular play mode"""
        return strategyData.play_mode == self.world.M_PLAY_ON


    def handle_special_game_modes(self, strategyData):
        """Handle kickoffs, kick-ins, etc."""
        drawer = self.world.draw
        
        kickoff_modes = [self.world.M_KICKOFF_LEFT, self.world.M_KICKOFF_RIGHT]
        kickin_modes = [self.world.M_KICK_IN_LEFT, self.world.M_KICK_IN_RIGHT]
        
        drawer.annotation((0, 10.5), f"Mode: {strategyData.play_mode}", drawer.Color.orange, "status")
        
        if strategyData.play_mode in kickoff_modes:
            return self.handle_kickoff(strategyData)
        elif strategyData.play_mode in kickin_modes:
            return self.handle_kickin(strategyData)
        else:
            return self.move(strategyData.my_desired_position)


    def handle_kickoff(self, strategyData):
        """Handle kickoff behavior"""
        if strategyData.am_i_closest_to_ball():
            if strategyData.can_i_kick():
                target = (15, 0)
                return self.kickTarget(strategyData, strategyData.mypos, target)
            else:
                return self.move_with_astar(
                    strategyData,
                    target_2d=strategyData.ball_2d,
                    avoid_opponents=False,
                    aggressive_mode=True
                )
        else:
            return self.move(strategyData.my_desired_position)


    def handle_kickin(self, strategyData):
        """Handle kick-in behavior"""
        if strategyData.am_i_closest_to_ball():
            if strategyData.can_i_kick():
                target, target_unum = strategyData.get_best_pass_target()
                self.world.draw.line(strategyData.mypos, target, 3, 
                                   self.world.draw.Color.red, "kickin_pass")
                return self.kickTarget(strategyData, strategyData.mypos, target)
            else:
                return self.move_with_astar(
                    strategyData,
                    target_2d=strategyData.ball_2d,
                    avoid_opponents=False,
                    aggressive_mode=True
                )
        else:
            return self.move(strategyData.my_desired_position)


    def play_soccer(self, strategyData):
        """Main gameplay logic during PlayOn mode"""
        drawer = self.world.draw
        
        if strategyData.am_i_closest_to_ball():
            drawer.annotation(strategyData.mypos, "ATTACKER", 
                            drawer.Color.red, f"role_{strategyData.player_unum}")
            return self.be_attacker(strategyData)
        else:
            drawer.annotation(strategyData.mypos, "SUPPORT", 
                            drawer.Color.blue, f"role_{strategyData.player_unum}")
            return self.be_supporter(strategyData)


    def be_attacker(self, strategyData):
        """Behavior when I'm going for the ball"""
        drawer = self.world.draw
        
        if strategyData.can_i_kick():
            drawer.annotation((0, 10.5), "KICKING", drawer.Color.yellow, "status")
            target = self.decide_kick_target(strategyData)
            drawer.line(strategyData.mypos, target, 3, drawer.Color.red, "kick_line")
            return self.kickTarget(strategyData, strategyData.mypos, target)
        else:
            drawer.annotation((0, 10.5), "CHASING BALL", drawer.Color.yellow, "status")
            return self.move_with_astar(
                strategyData,
                target_2d=strategyData.ball_2d,
                orientation=None,
                avoid_opponents=True,
                avoid_teammates=False,
                aggressive_mode=True
            )


    def be_supporter(self, strategyData):
        """Behavior when I'm NOT going for the ball"""
        drawer = self.world.draw
        drawer.clear(f"role_{strategyData.player_unum}")
        drawer.clear("kick_line")
        
        return self.move_with_astar(
            strategyData,
            target_2d=strategyData.my_desired_position,
            orientation=strategyData.ball_dir,
            avoid_opponents=True,
            avoid_teammates=True,
            aggressive_mode=False
        )


    def decide_kick_target(self, strategyData):
        """Decide whether to pass or shoot"""
        if strategyData.should_i_shoot():
            return (15, 0)
        elif strategyData.should_i_pass():
            pass_target, pass_unum = strategyData.get_best_pass_target()
            if pass_unum:
                self.world.draw.annotation(
                    pass_target, 
                    f"PASS TO {pass_unum}", 
                    self.world.draw.Color.cyan, 
                    "pass_target"
                )
            return pass_target
        else:
            return (15, 0)


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