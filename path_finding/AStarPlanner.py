import math
import heapq
from typing import List, Tuple, Optional

class AStarPlanner:
    """A* path planner for navigating around opponents"""
    
    def __init__(self, field_width=30, field_height=20, grid_resolution=0.5):
        """
        Args:
            field_width: Field width in meters (default: 30m, -15 to 15)
            field_height: Field height in meters (default: 20m, -10 to 10)
            grid_resolution: Size of each grid cell in meters
        """
        self.field_width = field_width
        self.field_height = field_height
        self.resolution = grid_resolution
        
        # Grid dimensions
        self.grid_width = int(field_width / grid_resolution)
        self.grid_height = int(field_height / grid_resolution)
        
        # Obstacle inflation radius (make obstacles appear larger for safety)
        self.robot_radius = 0.3  # meters
        self.obstacle_inflation = 0.5  # extra safety margin
        
    def world_to_grid(self, pos):
        """Convert world coordinates to grid indices"""
        # World: x in [-15, 15], y in [-10, 10]
        # Grid: indices [0, grid_width), [0, grid_height)
        grid_x = int((pos[0] + 15) / self.resolution)
        grid_y = int((pos[1] + 10) / self.resolution)
        
        # Clamp to valid range
        grid_x = max(0, min(self.grid_width - 1, grid_x))
        grid_y = max(0, min(self.grid_height - 1, grid_y))
        
        return (grid_x, grid_y)
    
    def grid_to_world(self, grid_pos):
        """Convert grid indices to world coordinates"""
        world_x = grid_pos[0] * self.resolution - 15
        world_y = grid_pos[1] * self.resolution - 10
        return (world_x, world_y)
    
    def create_obstacle_grid(self, obstacles, aggressive_mode=False):
        """
        Create a binary obstacle grid
        
        Args:
            obstacles: List of (x, y) positions of obstacles (opponents/teammates)
            aggressive_mode: If True, reduce safety margins
        """
        grid = [[0 for _ in range(self.grid_height)] for _ in range(self.grid_width)]
        
        inflation = self.obstacle_inflation if not aggressive_mode else 0.2
        
        for obstacle in obstacles:
            obs_grid = self.world_to_grid(obstacle)
            
            # Inflate obstacle (mark nearby cells as occupied)
            inflation_cells = int(inflation / self.resolution)
            
            for dx in range(-inflation_cells, inflation_cells + 1):
                for dy in range(-inflation_cells, inflation_cells + 1):
                    nx = obs_grid[0] + dx
                    ny = obs_grid[1] + dy
                    
                    # Check if within grid bounds
                    if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                        # Check if within circular radius
                        dist = math.sqrt(dx**2 + dy**2) * self.resolution
                        if dist <= inflation + self.robot_radius:
                            grid[nx][ny] = 1  # Mark as obstacle
        
        return grid
    
    def heuristic(self, pos1, pos2):
        """Euclidean distance heuristic"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def get_neighbors(self, pos, grid):
        """Get valid neighboring cells (8-connected grid)"""
        neighbors = []
        
        # 8 directions: N, S, E, W, NE, NW, SE, SW
        directions = [
            (0, 1), (0, -1), (1, 0), (-1, 0),  # Cardinal
            (1, 1), (1, -1), (-1, 1), (-1, -1)  # Diagonal
        ]
        
        for dx, dy in directions:
            nx = pos[0] + dx
            ny = pos[1] + dy
            
            # Check bounds
            if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                # Check if not obstacle
                if grid[nx][ny] == 0:
                    # Cost is higher for diagonal moves
                    cost = 1.414 if dx != 0 and dy != 0 else 1.0
                    neighbors.append(((nx, ny), cost))
        
        return neighbors
    
    def reconstruct_path(self, came_from, start, goal):
        """Reconstruct path from start to goal using came_from dict"""
        path = []
        current = goal
        
        while current != start:
            path.append(current)
            current = came_from.get(current)
            if current is None:
                return []  # Path reconstruction failed
        
        path.append(start)
        path.reverse()
        
        # Convert to world coordinates
        world_path = [self.grid_to_world(p) for p in path]
        
        return world_path
    
    def smooth_path(self, path):
        """Smooth path by removing unnecessary waypoints"""
        if len(path) <= 2:
            return path
        
        smoothed = [path[0]]
        
        i = 0
        while i < len(path) - 1:
            # Look ahead to find furthest visible point
            for j in range(len(path) - 1, i, -1):
                if self.is_line_clear(path[i], path[j]):
                    smoothed.append(path[j])
                    i = j
                    break
            else:
                i += 1
        
        return smoothed
    
    def is_line_clear(self, p1, p2):
        """Check if line between two points is obstacle-free (simplified)"""
        # For now, assume clear. You can add line-obstacle intersection checks
        return True
    
    def plan_path(self, start_pos, goal_pos, obstacles, aggressive_mode=False, max_iterations=1000):
        """
        Plan path using A* algorithm
        
        Args:
            start_pos: (x, y) starting position in world coordinates
            goal_pos: (x, y) goal position in world coordinates
            obstacles: List of (x, y) obstacle positions
            aggressive_mode: If True, reduce safety margins
            max_iterations: Maximum search iterations
            
        Returns:
            path: List of (x, y) waypoints in world coordinates
            success: Boolean indicating if path was found
        """
        # Convert to grid coordinates
        start_grid = self.world_to_grid(start_pos)
        goal_grid = self.world_to_grid(goal_pos)
        
        # Create obstacle grid
        grid = self.create_obstacle_grid(obstacles, aggressive_mode)
        
        # Check if start or goal is in obstacle
        if grid[start_grid[0]][start_grid[1]] == 1:
            return [goal_pos], False  # Can't plan from obstacle, return direct path
        
        if grid[goal_grid[0]][goal_grid[1]] == 1:
            # Goal is blocked, find nearest free cell
            goal_grid = self.find_nearest_free_cell(goal_grid, grid)
        
        # A* algorithm
        open_set = []
        heapq.heappush(open_set, (0, start_grid))
        
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self.heuristic(start_grid, goal_grid)}
        
        closed_set = set()
        iterations = 0
        
        while open_set and iterations < max_iterations:
            iterations += 1
            
            current_f, current = heapq.heappop(open_set)
            
            if current == goal_grid:
                # Path found!
                path = self.reconstruct_path(came_from, start_grid, goal_grid)
                smoothed_path = self.smooth_path(path)
                return smoothed_path, True
            
            closed_set.add(current)
            
            for neighbor, move_cost in self.get_neighbors(current, grid):
                if neighbor in closed_set:
                    continue
                
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal_grid)
                    
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # No path found, return direct path
        return [start_pos, goal_pos], False
    
    def find_nearest_free_cell(self, pos, grid):
        """Find nearest unoccupied cell to given position"""
        visited = set()
        queue = [pos]
        
        while queue:
            current = queue.pop(0)
            
            if current in visited:
                continue
            visited.add(current)
            
            if grid[current[0]][current[1]] == 0:
                return current
            
            # Add neighbors
            for (nx, ny), _ in self.get_neighbors(current, [[0]*self.grid_height for _ in range(self.grid_width)]):
                if (nx, ny) not in visited:
                    queue.append((nx, ny))
        
        return pos  # Fallback