import pygame
import random
import math
import numpy as np
from collections import defaultdict, deque
import pickle
import os


pygame.init()


WIDTH, HEIGHT = 1200, 800
PLAY_AREA = (WIDTH - 300, HEIGHT)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
EXIT_RADIUS = 15
EXIT_POS = (PLAY_AREA[0] // 2, 50)


BALL_RADIUS = 8
BALL_SPEED = 2
STUCK_DISTANCE = 40
STUCK_FRAMES = 120


LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EXPLORATION_RATE = 1.0
MIN_EXPLORATION_RATE = 0.01
EXPLORATION_DECAY = 0.995
GRID_SIZE = 20  # Размер ячейки для Q-таблицы
Q_TABLE_FILE = "qtable.pkl"


BALL_COLORS = {
    "синий": (49, 8, 255),
    "оранж": (255, 144, 8),
    "фукси": (222, 8, 255)
}

TRAIL_COLORS = {
    "синий": (181, 197, 255),
    "оранж": (252, 230, 192),
    "фукси": (255, 189, 254)
}


INITIAL_POSITIONS = {
    "синий": (100, HEIGHT - 100),
    "фукси": (PLAY_AREA[0] // 2, HEIGHT - 100),
    "оранж": (PLAY_AREA[0] - 100, HEIGHT - 100)
}


HUNTER_RADIUS = 10
HUNTER_SPEED = 3.5
HUNTER_COLOR = BLACK
HUNTER_PATIENCE = 120


OBSTACLE_MEMORY_RADIUS = 50
OBSTACLE_AVOIDANCE_FACTOR = 0.2
PATH_DEVIATION_ANGLE = math.pi / 4


screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Интеллектуальный поиск пути шариками с Q-обучением")
clock = pygame.time.Clock()
font = pygame.font.SysFont('Arial', 16)
small_font = pygame.font.SysFont('Arial', 14)

class QLearning:
    def __init__(self, actions):
        self.actions = actions  # Возможные действия (углы движения)
        self.q_table = defaultdict(lambda: np.zeros(len(actions)))
        self.exploration_rate = EXPLORATION_RATE
        self.learning_rate = LEARNING_RATE
        self.discount_factor = DISCOUNT_FACTOR
        

        self.load_q_table()
    
    def get_state_key(self, x, y, hunter_dist, exit_dist):

        grid_x = int(x // GRID_SIZE)
        grid_y = int(y // GRID_SIZE)
        hunter_dist_key = min(int(hunter_dist // 50), 4) if hunter_dist < 200 else 4
        exit_dist_key = min(int(exit_dist // 50), 4) if exit_dist < 200 else 4
        return (grid_x, grid_y, hunter_dist_key, exit_dist_key)
    
    def choose_action(self, state_key):

        if random.random() < self.exploration_rate:
            return random.choice(self.actions)
        else:
            return self.actions[np.argmax(self.q_table[state_key])]
    
    def learn(self, state, action_idx, reward, next_state):

        current_q = self.q_table[state][action_idx]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action_idx] = new_q
    
    def decay_exploration(self):
        self.exploration_rate = max(MIN_EXPLORATION_RATE, self.exploration_rate * EXPLORATION_DECAY)
    
    def save_q_table(self):
        with open(Q_TABLE_FILE, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
    
    def load_q_table(self):
        if os.path.exists(Q_TABLE_FILE):
            with open(Q_TABLE_FILE, 'rb') as f:
                loaded = pickle.load(f)
                self.q_table.update(loaded)
            print(f"Загружена Q-таблица с {len(self.q_table)} состояниями")

class Obstacle:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
        self.center = (x + width // 2, y + height // 2)
        self.width = width
        self.height = height
    
    def draw(self):
        pygame.draw.rect(screen, RED, self.rect)

class Hunter:
    def __init__(self):
        self.reset()
        self.manual_control = False
    
    def reset(self):
        self.x = random.randint(HUNTER_RADIUS, PLAY_AREA[0] - HUNTER_RADIUS)
        self.y = random.randint(HUNTER_RADIUS, HEIGHT - HUNTER_RADIUS)
        self.speed = HUNTER_SPEED
        self.kills = 0
        self.current_target = None
        self.target_timer = 0
        self.last_killed_color = None
        self.color_kills = {"синий": 0, "оранж": 0, "фукси": 0}
    
    def move(self, balls, keys=None):
        if self.manual_control:
            self.manual_move(keys)
            return
        
        if self.current_target is None or self.target_timer <= 0:
            self.choose_new_target(balls)
        
        self.target_timer -= 1
        
        if self.current_target:
            angle = math.atan2(self.current_target.y - self.y, 
                              self.current_target.x - self.x)
            self.x += self.speed * math.cos(angle)
            self.y += self.speed * math.sin(angle)
        

        self.x = max(HUNTER_RADIUS, min(PLAY_AREA[0] - HUNTER_RADIUS, self.x))
        self.y = max(HUNTER_RADIUS, min(HEIGHT - HUNTER_RADIUS, self.y))
    
    def manual_move(self, keys):
        if keys[pygame.K_w]:
            self.y -= self.speed
        if keys[pygame.K_s]:
            self.y += self.speed
        if keys[pygame.K_a]:
            self.x -= self.speed
        if keys[pygame.K_d]:
            self.x += self.speed
        
        self.x = max(HUNTER_RADIUS, min(PLAY_AREA[0] - HUNTER_RADIUS, self.x))
        self.y = max(HUNTER_RADIUS, min(HEIGHT - HUNTER_RADIUS, self.y))
    
    def choose_new_target(self, balls):
        available_balls = [ball for ball in balls 
                         if not ball.reached_exit and not ball.knows_exact_path]
        
        if self.last_killed_color and len(available_balls) > 1:
            available_balls = [ball for ball in available_balls 
                             if ball.color_name != self.last_killed_color]
        
        min_kills = min(self.color_kills.values())
        candidates = [color for color, count in self.color_kills.items() 
                     if count <= min_kills + 2]
        
        if len(candidates) > 0:
            available_balls = [ball for ball in available_balls 
                             if ball.color_name in candidates]
        
        if available_balls:
            self.current_target = min(
                available_balls,
                key=lambda ball: math.hypot(self.x - ball.x, self.y - ball.y)
            )
            self.target_timer = HUNTER_PATIENCE
        else:
            self.current_target = None
    
    def draw(self):
        pygame.draw.circle(screen, HUNTER_COLOR, (int(self.x), int(self.y)), HUNTER_RADIUS)
        
        if self.current_target and not self.manual_control:
            pygame.draw.line(screen, (255, 0, 0), 
                           (self.x, self.y),
                           (self.current_target.x, self.current_target.y), 1)
    
    def check_collision(self, balls):
        for i, ball in enumerate(balls):
            if (not ball.reached_exit and not ball.knows_exact_path and 
                math.hypot(self.x - ball.x, self.y - ball.y) < HUNTER_RADIUS + BALL_RADIUS):
                self.last_killed_color = balls[i].color_name
                self.color_kills[balls[i].color_name] += 1
                balls[i] = ball.die()
                self.kills += 1
                self.choose_new_target([b for b in balls if not b.reached_exit and not b.knows_exact_path])
                return True
        return False

class Ball:
    all_trails = defaultdict(list)
    obstacle_knowledge = defaultdict(dict)
    successful_paths = defaultdict(list)
    q_learners = {}  # Отдельные Q-обучатели для каждого цвета
    
    def __init__(self, color_name, generation=1, initial_pos=None):
        self.color_name = color_name
        self.color = BALL_COLORS[color_name]
        self.trail_color = TRAIL_COLORS[color_name]
        self.initial_pos = initial_pos if initial_pos else INITIAL_POSITIONS.get(color_name, (PLAY_AREA[0]//2, HEIGHT//2))
        self.x, self.y = self.initial_pos
        self.radius = BALL_RADIUS
        self.speed = BALL_SPEED
        self.generation = generation
        self.escapes = 0
        self.memory = []
        self.trail = []
        

        if color_name not in Ball.q_learners:

            actions = [i * math.pi/4 for i in range(8)]
            Ball.q_learners[color_name] = QLearning(actions)
        
        self.q_learner = Ball.q_learners[color_name]
        self.current_action_idx = None
        self.last_state = None
        self.last_action_idx = None
        
        self.angle = random.uniform(0, 2 * math.pi)
        self.immortal = False
        self.immortal_timer = 0
        self.vision_radius = 0
        self.see_obstacles = False
        self.see_hunter = False
        self.ancestor_deaths = defaultdict(int)
        self.ancestor_trails = []
        self.knows_hunter = generation >= random.randint(5, 15)
        self.last_positions = []
        self.stuck_timer = 0
        self.knows_exact_path = False
        self.reached_exit = False
        self.explored_cells = set()
        self.total_cells = (PLAY_AREA[0]//20) * (HEIGHT//20)
        self.obstacle_memory = defaultdict(int)
        self.area_history = []
        self.max_area_history = STUCK_FRAMES
        self.exploration_boost = 1.0
        self.obstacle_avoidance = defaultdict(float)
        self.path_memory = []
        self.obstacle_map = defaultdict(list)
        self.path_finding_attempts = 0
        self.current_obstacle = None
        self.obstacle_circuit_direction = 0
        self.obstacle_circuit_progress = 0
        self.last_exit_distance = float('inf')
        self.successful_path = []
        self.boundary_bounce_count = 0
        self.last_reward = 0
        self.total_reward = 0
        
        if color_name in Ball.obstacle_knowledge:
            self.obstacle_map = Ball.obstacle_knowledge[color_name].copy()
        
        if generation >= 20:
            self.see_obstacles = True
            self.vision_radius = 50 + 10 * ((generation - 20) // 10)
    
    def update(self, obstacles, hunter, balls):
        if self.reached_exit:
            return False
        
        cell_x, cell_y = int(self.x)//20, int(self.y)//20
        self.explored_cells.add((cell_x, cell_y))
        
        self.area_history.append((self.x, self.y))
        if len(self.area_history) > self.max_area_history:
            self.area_history.pop(0)
        
        if self.check_stuck():
            self.last_reward = -10  # Наказание за застревание
            return True
        
        if self.immortal:
            self.immortal_timer -= 1
            if self.immortal_timer <= 0:
                self.immortal = False
        
        ball_rect = pygame.Rect(self.x - self.radius, self.y - self.radius, 
                               self.radius*2, self.radius*2)
        for obstacle in obstacles:
            if ball_rect.colliderect(obstacle.rect) and not self.immortal:
                self.remember_obstacle(obstacle)
                self.last_reward = -20  # Большое наказание за столкновение с препятствием
                return True
        
        self.handle_hunter(hunter)
        

        hunter_dist = math.hypot(self.x - hunter.x, self.y - hunter.y)
        exit_dist = math.hypot(EXIT_POS[0] - self.x, EXIT_POS[1] - self.y)
        state = self.q_learner.get_state_key(self.x, self.y, hunter_dist, exit_dist)
        

        if self.generation > 5:  # Начинаем использовать Q-обучение после 5 поколений
            action = self.q_learner.choose_action(state)
            self.angle = action
            self.current_action_idx = self.q_learner.actions.index(action)
        else:

            self.angle += random.uniform(-0.5, 0.5)
            self.current_action_idx = None
        

        if self.last_state is not None and self.last_action_idx is not None and self.generation > 5:

            reward = self.calculate_reward(exit_dist, hunter_dist)
            self.total_reward += reward
            

            self.q_learner.learn(self.last_state, self.last_action_idx, reward, state)
        
        self.last_state = state
        self.last_action_idx = self.current_action_idx
        
        if self.knows_exact_path:
            self.smart_move_to_exit(obstacles)
        else:
            self.explore_environment(obstacles)
        

        if self.x < self.radius:
            self.x = self.radius
            self.angle = math.pi - self.angle + random.uniform(-0.2, 0.2)
            self.boundary_bounce_count += 1
            self.last_reward = -2  # Небольшое наказание за выход за границы
        elif self.x > PLAY_AREA[0] - self.radius:
            self.x = PLAY_AREA[0] - self.radius
            self.angle = math.pi - self.angle + random.uniform(-0.2, 0.2)
            self.boundary_bounce_count += 1
            self.last_reward = -2
        if self.y < self.radius:
            self.y = self.radius
            self.angle = -self.angle + random.uniform(-0.2, 0.2)
            self.boundary_bounce_count += 1
            self.last_reward = -2
        elif self.y > HEIGHT - self.radius:
            self.y = HEIGHT - self.radius
            self.angle = -self.angle + random.uniform(-0.2, 0.2)
            self.boundary_bounce_count += 1
            self.last_reward = -2
        

        if self.boundary_bounce_count > 5:
            self.angle = random.uniform(0, 2 * math.pi)
            self.boundary_bounce_count = 0
        
        exit_dist = math.hypot(EXIT_POS[0] - self.x, EXIT_POS[1] - self.y)
        if exit_dist < EXIT_RADIUS + self.radius:
            self.successful_path = self.trail.copy()
            Ball.successful_paths[self.color_name].append(self.successful_path)
            self.reached_exit = True
            self.last_reward = 50  # Большая награда за достижение выхода
            return True
        
        self.add_trail()
        
        return False
    
    def calculate_reward(self, exit_dist, hunter_dist):
        reward = 0
        

        if exit_dist < self.last_exit_distance:
            reward += 1
        else:
            reward -= 1
        
        self.last_exit_distance = exit_dist
        

        cell_x, cell_y = int(self.x)//20, int(self.y)//20
        if (cell_x, cell_y) not in self.explored_cells:
            reward += 0.5
        

        if hunter_dist < 100 and self.knows_hunter:
            reward -= 2
        
        return reward
    
    def smart_move_to_exit(self, obstacles):
        exit_dir_x = EXIT_POS[0] - self.x
        exit_dir_y = EXIT_POS[1] - self.y
        exit_dist = math.hypot(exit_dir_x, exit_dir_y)
        self.last_exit_distance = exit_dist
        
        if exit_dist > 0:
            exit_dir_x /= exit_dist
            exit_dir_y /= exit_dist
            

            closest_obstacle = None
            min_obstacle_dist = float('inf')
            obstacle_blocking = False
            
            for obstacle in obstacles:
                if self.line_intersects_obstacle(self.x, self.y, 
                                               EXIT_POS[0], EXIT_POS[1], 
                                               obstacle):
                    dist = math.hypot(obstacle.rect.centerx - self.x, 
                                     obstacle.rect.centery - self.y)
                    if dist < min_obstacle_dist:
                        min_obstacle_dist = dist
                        closest_obstacle = obstacle
                    obstacle_blocking = True
            

            for pos, data in self.obstacle_map.items():
                for obstacle_type, size in data:
                    if self.line_intersects_point(self.x, self.y, 
                                                EXIT_POS[0], EXIT_POS[1], 
                                                pos[0], pos[1], size):
                        obstacle_blocking = True
                        if pos[0] < min_obstacle_dist:
                            min_obstacle_dist = pos[0]
            
            if obstacle_blocking and closest_obstacle:

                self.avoid_obstacle(closest_obstacle, exit_dir_x, exit_dir_y)
            else:

                target_angle = math.atan2(exit_dir_y, exit_dir_x)
                

                angle_diff = (target_angle - self.angle + math.pi) % (2 * math.pi) - math.pi
                self.angle += angle_diff * 0.3
                

                if self.generation > 50:

                    avoid_x, avoid_y = 0, 0
                    for spot, count in self.ancestor_deaths.items():
                        dist = math.hypot(self.x - spot[0], self.y - spot[1])
                        if dist < self.vision_radius:
                            weight = count / (dist + 1) * 0.5
                            avoid_x += weight * (self.x - spot[0])
                            avoid_y += weight * (self.y - spot[1])
                    
                    if avoid_x != 0 or avoid_y != 0:
                        avoid_angle = math.atan2(avoid_y, avoid_x)
                        angle_diff = (avoid_angle - self.angle + math.pi) % (2 * math.pi) - math.pi
                        self.angle += angle_diff * 0.1
                
                self.current_obstacle = None
                self.obstacle_circuit_direction = 0
        else:
            self.angle = random.uniform(0, 2 * math.pi)
        

        self.x += self.speed * math.cos(self.angle)
        self.y += self.speed * math.sin(self.angle)
    
    def avoid_obstacle(self, obstacle, exit_dir_x, exit_dir_y):
        if self.current_obstacle != obstacle:
            self.current_obstacle = obstacle
            self.obstacle_circuit_progress = 0
            

            if self.generation > 30:

                left_deaths = 0
                right_deaths = 0
                
                for spot, count in self.ancestor_deaths.items():
                    dist = math.hypot(obstacle.rect.centerx - spot[0], 
                                     obstacle.rect.centery - spot[1])
                    if dist < self.vision_radius:
                        angle = math.atan2(spot[1] - obstacle.rect.centery, 
                                         spot[0] - obstacle.rect.centerx)
                        exit_angle = math.atan2(exit_dir_y, exit_dir_x)
                        angle_diff = (angle - exit_angle + math.pi) % (2 * math.pi) - math.pi
                        
                        if angle_diff > 0:
                            right_deaths += count
                        else:
                            left_deaths += count
                

                if left_deaths < right_deaths:
                    self.obstacle_circuit_direction = -1
                else:
                    self.obstacle_circuit_direction = 1
            else:
                self.obstacle_circuit_direction = random.choice([-1, 1])
        

        if obstacle.width > obstacle.height:  # Горизонтальное препятствие
            if self.obstacle_circuit_direction == 1:
                target_x = obstacle.rect.right + 30 + random.uniform(-10, 10)
                target_y = obstacle.rect.centery + random.uniform(-10, 10)
            else:
                target_x = obstacle.rect.left - 30 + random.uniform(-10, 10)
                target_y = obstacle.rect.centery + random.uniform(-10, 10)
        else:  # Вертикальное препятствие
            if self.obstacle_circuit_direction == 1:
                target_x = obstacle.rect.centerx + random.uniform(-10, 10)
                target_y = obstacle.rect.bottom + 30 + random.uniform(-10, 10)
            else:
                target_x = obstacle.rect.centerx + random.uniform(-10, 10)
                target_y = obstacle.rect.top - 30 + random.uniform(-10, 10)
        

        dir_x = target_x - self.x
        dir_y = target_y - self.y
        dist = math.hypot(dir_x, dir_y)
        
        if dist > 0:
            dir_x /= dist
            dir_y /= dist
            
            target_angle = math.atan2(dir_y, dir_x)
            angle_diff = (target_angle - self.angle + math.pi) % (2 * math.pi) - math.pi
            

            if self.generation > 50:
                self.angle += angle_diff * 0.5
            else:
                self.angle += angle_diff * 0.3
            

            if self.generation < 30:
                self.angle += random.uniform(-0.2, 0.2)
        

        if dist < 15:
            self.obstacle_circuit_progress += 1
            
            if self.obstacle_circuit_progress > 1:
                self.current_obstacle = None
                self.obstacle_circuit_direction = 0
    
    def explore_environment(self, obstacles):
        exit_dist = math.hypot(EXIT_POS[0] - self.x, EXIT_POS[1] - self.y)
        if exit_dist < self.vision_radius * 2:
            self.knows_exact_path = True
            self.path_memory.append((self.x, self.y))
            return
        

        if random.random() < 0.1 or len(self.explored_cells) < 10:
            directions = []
            

            if self.generation > 30:
                safe_directions = []
                danger_directions = []
                
                for dir_angle in [i * math.pi/4 for i in range(8)]:
                    test_x = self.x + math.cos(dir_angle) * 100
                    test_y = self.y + math.sin(dir_angle) * 100
                    test_cell = (int(test_x)//20, int(test_y)//20)
                    

                    danger_score = 0
                    for spot, count in self.ancestor_deaths.items():
                        dist = math.hypot(test_x - spot[0], test_y - spot[1])
                        if dist < self.vision_radius:
                            danger_score += count / (dist + 1)
                    
                    if danger_score < 1:  # Безопасное направление
                        safe_directions.append(dir_angle)
                    else:
                        danger_directions.append((dir_angle, danger_score))
                
                if safe_directions:
                    directions = safe_directions
                else:

                    danger_directions.sort(key=lambda x: x[1])
                    directions = [x[0] for x in danger_directions[:3]]
            else:
                directions = [i * math.pi/4 for i in range(8)]
            

            least_explored_dir = 0
            min_explored = float('inf')
            
            for dir_angle in directions:
                test_x = self.x + math.cos(dir_angle) * 100
                test_y = self.y + math.sin(dir_angle) * 100
                test_cell = (int(test_x)//20, int(test_y)//20)
                
                explored_count = sum(1 for cell in self.explored_cells 
                                   if math.hypot(cell[0]-test_cell[0], cell[1]-test_cell[1]) < 3)
                
                if explored_count < min_explored:
                    min_explored = explored_count
                    least_explored_dir = dir_angle
            

            angle_diff = (least_explored_dir - self.angle + math.pi) % (2 * math.pi) - math.pi
            self.angle += angle_diff * 0.3
            

            if self.generation < 10:
                self.angle += random.uniform(-0.5, 0.5)
            elif self.generation < 20:
                self.angle += random.uniform(-0.3, 0.3)
            else:
                self.angle += random.uniform(-0.1, 0.1)
        

        self.x += self.speed * math.cos(self.angle)
        self.y += self.speed * math.sin(self.angle)
    
    def check_stuck(self):
        if len(self.area_history) == self.max_area_history:
            min_x = min(p[0] for p in self.area_history)
            max_x = max(p[0] for p in self.area_history)
            min_y = min(p[1] for p in self.area_history)
            max_y = max(p[1] for p in self.area_history)
            
            if max_x - min_x < STUCK_DISTANCE and max_y - min_y < STUCK_DISTANCE:
                return True
        return False
    
    def remember_obstacle(self, obstacle):
        key = (obstacle.rect.x, obstacle.rect.y)
        self.obstacle_memory[key] += 1
        self.obstacle_avoidance[key] += OBSTACLE_AVOIDANCE_FACTOR
        
        if obstacle.width > obstacle.height:  # Горизонтальное препятствие
            for x in range(obstacle.rect.left, obstacle.rect.right, 5):
                self.obstacle_map[(x, obstacle.rect.centery)].append(("horizontal", obstacle.height))
        else:  # Вертикальное препятствие
            for y in range(obstacle.rect.top, obstacle.rect.bottom, 5):
                self.obstacle_map[(obstacle.rect.centerx, y)].append(("vertical", obstacle.width))
        
        Ball.obstacle_knowledge[self.color_name] = self.obstacle_map
    
    def handle_hunter(self, hunter):
        hunter_dist = math.hypot(self.x - hunter.x, self.y - hunter.y)
        if self.knows_hunter and hunter_dist < self.vision_radius + 50:
            self.see_hunter = True
            self.speed = BALL_SPEED * 1.5
        else:
            self.see_hunter = False
            self.speed = BALL_SPEED
    
    def line_intersects_obstacle(self, x1, y1, x2, y2, obstacle):
        return self.line_intersects_rect(x1, y1, x2, y2, obstacle.rect)
    
    def line_intersects_rect(self, x1, y1, x2, y2, rect):
        left = self.line_intersects_line(x1, y1, x2, y2, rect.left, rect.top, rect.left, rect.bottom)
        right = self.line_intersects_line(x1, y1, x2, y2, rect.right, rect.top, rect.right, rect.bottom)
        top = self.line_intersects_line(x1, y1, x2, y2, rect.left, rect.top, rect.right, rect.top)
        bottom = self.line_intersects_line(x1, y1, x2, y2, rect.left, rect.bottom, rect.right, rect.bottom)
        
        return left or right or top or bottom
    
    def line_intersects_point(self, x1, y1, x2, y2, px, py, size):

        min_x = min(x1, x2)
        max_x = max(x1, x2)
        min_y = min(y1, y2)
        max_y = max(y1, y2)
        
        if not (min_x - size <= px <= max_x + size and min_y - size <= py <= max_y + size):
            return False
        

        if x1 == x2:  # Вертикальная линия
            dist = abs(px - x1)
        elif y1 == y2:  # Горизонтальная линия
            dist = abs(py - y1)
        else:
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            dist = abs(A * px + B * py + C) / math.sqrt(A*A + B*B)
        
        return dist <= size
    
    def line_intersects_line(self, x1, y1, x2, y2, x3, y3, x4, y4):
        def ccw(A, B, C):
            return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
        
        A = (x1, y1)
        B = (x2, y2)
        C = (x3, y3)
        D = (x4, y4)
        
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
    
    def add_trail(self):
        self.trail.append((self.x, self.y))
        if len(self.trail) > 1000:
            self.trail.pop(0)
    
    def get_explored_percent(self):
        return len(self.explored_cells) / self.total_cells * 100
    
    def die(self):
        death_spot = (int(self.x), int(self.y))
        self.memory.append(("death", death_spot))
        

        if self.last_state is not None and self.last_action_idx is not None and self.generation > 5:
            self.q_learner.learn(self.last_state, self.last_action_idx, -20, None)
        
        return self.respawn()
    
    def escape(self):
        self.escapes += 1
        self.memory.append(("escape", (int(self.x), int(self.y))))
        

        if self.last_state is not None and self.last_action_idx is not None and self.generation > 5:
            self.q_learner.learn(self.last_state, self.last_action_idx, 50, None)
        
        return self.respawn()
    
    def respawn(self):
        new_ball = Ball(self.color_name, self.generation + 1, self.initial_pos)
        new_ball.ancestor_deaths = self.ancestor_deaths.copy()
        new_ball.ancestor_trails = self.ancestor_trails.copy()
        new_ball.escapes = self.escapes
        new_ball.knows_exact_path = self.knows_exact_path
        new_ball.explored_cells = self.explored_cells.copy()
        new_ball.obstacle_memory = self.obstacle_memory.copy()
        new_ball.obstacle_avoidance = self.obstacle_avoidance.copy()
        new_ball.path_memory = self.path_memory.copy()
        new_ball.obstacle_map = self.obstacle_map.copy()
        new_ball.last_exit_distance = self.last_exit_distance
        
        for memory in self.memory:
            if memory[0] == "death":
                new_ball.ancestor_deaths[memory[1]] += 1
        
        if len(self.trail) > 0:
            Ball.all_trails[self.color_name].append(self.trail.copy())
            new_ball.ancestor_trails.append(self.trail.copy())
        
        new_ball.immortal = True
        new_ball.immortal_timer = 60
        

        self.q_learner.decay_exploration()
        
        return new_ball
    
    def draw(self):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)
        
        if self.immortal:
            pygame.draw.circle(screen, (200, 200, 200), (int(self.x), int(self.y)), 
                             self.radius + 2, 1)
        
        if self.see_hunter:
            pygame.draw.circle(screen, (255, 100, 100), (int(self.x), int(self.y)), 
                             self.radius + 4, 1)

def create_obstacles():
    obstacles = []
    

    obstacles.append(Obstacle(EXIT_POS[0] - 50, 0, 30, 80))
    obstacles.append(Obstacle(EXIT_POS[0] + 20, 0, 30, 80))
    

    for _ in range(10):
        if random.random() < 0.5:
            x = random.randint(0, PLAY_AREA[0] - 100)
            y = random.randint(100, HEIGHT - 100)
            width = random.randint(50, 200)
            height = random.randint(2, 5)
        else:
            x = random.randint(100, PLAY_AREA[0] - 100)
            y = random.randint(0, HEIGHT - 100)
            width = random.randint(2, 5)
            height = random.randint(50, 200)
        obstacles.append(Obstacle(x, y, width, height))
    
    return obstacles

def draw_stats(balls, hunter, speed):
    stats_x = PLAY_AREA[0] + 10
    stats_y = 20
    panel_width = WIDTH - PLAY_AREA[0] - 20
    
    pygame.draw.rect(screen, (240, 240, 240), (stats_x - 5, stats_y - 5, panel_width, HEIGHT - 30))
    pygame.draw.rect(screen, (200, 200, 200), (stats_x - 5, stats_y - 5, panel_width, HEIGHT - 30), 1)
    
    speed_text = font.render(f"Скорость: {speed:.1f}x", True, BLACK)
    screen.blit(speed_text, (stats_x, stats_y))
    controls_text = small_font.render("вверх/вниз", True, (100, 100, 100))
    screen.blit(controls_text, (stats_x + 100, stats_y + 2))
    stats_y += 30
    
    hunter_title = font.render("Охотник:", True, BLACK)
    screen.blit(hunter_title, (stats_x, stats_y))
    stats_y += 25
    
    kills_text = font.render(f"Убийств: {hunter.kills}", True, BLACK)
    screen.blit(kills_text, (stats_x + 10, stats_y))
    stats_y += 25
    
    deaths_title = font.render("Смерти:", True, BLACK)
    screen.blit(deaths_title, (stats_x, stats_y))
    stats_y += 25
    
    for color, kills in hunter.color_kills.items():
        color_text = font.render(f"{color}: {kills}", True, BALL_COLORS[color])
        screen.blit(color_text, (stats_x + 20, stats_y))
        stats_y += 20
    
    stats_y += 10
    
    control_title = font.render("Управление:", True, BLACK)
    screen.blit(control_title, (stats_x, stats_y))
    stats_y += 25
    
    mode_text = font.render(f"Режим: {'РУЧНОЙ' if hunter.manual_control else 'АВТО'}", 
                          True, (0, 100, 0) if hunter.manual_control else BLACK)
    screen.blit(mode_text, (stats_x + 10, stats_y))
    stats_y += 20
    
    controls = [
        ("M", "управленеи охотником"),
        ("Q", "пропуск сбора данных"),
        ("R", "перезапуск"),
        ("S", "сохранить Q-таблицу")
    ]
    
    for key, desc in controls:
        key_text = small_font.render(f"[{key}]", True, (0, 0, 150))
        desc_text = small_font.render(desc, True, (100, 100, 100))
        screen.blit(key_text, (stats_x + 10, stats_y))
        screen.blit(desc_text, (stats_x + 40, stats_y))
        stats_y += 20
    
    stats_y += 10
    
    balls_title = font.render("Шарики:", True, BLACK)
    screen.blit(balls_title, (stats_x, stats_y))
    stats_y += 25
    
    for ball in balls:
        ball_text = font.render(f"{ball.color_name}:", True, ball.color)
        gen_text = small_font.render(f"Ген {ball.generation}", True, BLACK)
        
        screen.blit(ball_text, (stats_x, stats_y))
        screen.blit(gen_text, (stats_x + 100, stats_y))
        stats_y += 20
        
        pos_text = small_font.render(f"Позиция: ({int(ball.x)}, {int(ball.y)})", True, (50, 50, 50))
        screen.blit(pos_text, (stats_x + 10, stats_y))
        stats_y += 20
        
        explored_text = small_font.render(
            f"Исследовано: {ball.get_explored_percent():.1f}%", 
            True, (50, 50, 50)
        )
        screen.blit(explored_text, (stats_x + 10, stats_y))
        stats_y += 20
        

        if ball.generation > 5:
            q_info = small_font.render(
                f"Q-обучение: ε={ball.q_learner.exploration_rate:.2f}", 
                True, (50, 50, 50)
            )
            screen.blit(q_info, (stats_x + 10, stats_y))
            stats_y += 20
        
        abilities = []
        if ball.see_obstacles:
            abilities.append("зрение")
        if ball.knows_hunter:
            abilities.append("знает охотника")
        if ball.knows_exact_path:
            abilities.append("ищет выход")
        if ball.reached_exit:
            abilities.append("УМНИЧКА")
        
        if abilities:
            abilities_text = small_font.render(", ".join(abilities), True, (50, 50, 50))
            screen.blit(abilities_text, (stats_x + 10, stats_y))
            stats_y += 20
        
        stats_y += 5

def check_victory(balls):
    return all(ball.reached_exit for ball in balls)

def save_all_q_tables():
    for color, q_learner in Ball.q_learners.items():
        q_learner.save_q_table()
    print("Все Q-таблицы сохранены")

def main():
    running = True
    speed = 1.0
    game_over = False
    
    balls = [
        Ball("синий"),
        Ball("фукси"),
        Ball("оранж")
    ]
    
    hunter = Hunter()
    obstacles = create_obstacles()
    
    while running:
        keys = pygame.key.get_pressed()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    Ball.all_trails.clear()
                    Ball.obstacle_knowledge.clear()
                    Ball.successful_paths.clear()
                    balls = [
                        Ball("синий"),
                        Ball("фукси"),
                        Ball("оранж")
                    ]
                    hunter = Hunter()
                    obstacles = create_obstacles()
                    game_over = False
                elif event.key == pygame.K_m:
                    hunter.manual_control = not hunter.manual_control
                elif event.key == pygame.K_q:
                    for ball in balls:
                        ball.knows_exact_path = True
                elif event.key == pygame.K_s:
                    save_all_q_tables()
        
        if not game_over:
            if keys[pygame.K_UP]:
                speed = min(speed + 0.05, 5.0)
            if keys[pygame.K_DOWN]:
                speed = max(speed - 0.05, 0.1)
            
            for _ in range(int(speed)):
                hunter.move(balls, keys)
                hunter.check_collision(balls)
                
                new_balls = []
                for ball in balls:
                    should_die = ball.update(obstacles, hunter, balls)
                    
                    if should_die and not ball.reached_exit:
                        new_balls.append(ball.die())
                    else:
                        new_balls.append(ball)
                
                balls = new_balls
            
            if check_victory(balls):
                game_over = True
        
        screen.fill(WHITE)
        
        pygame.draw.rect(screen, (240, 240, 240), (0, 0, PLAY_AREA[0], HEIGHT))
        pygame.draw.line(screen, BLACK, (PLAY_AREA[0], 0), (PLAY_AREA[0], HEIGHT), 2)
        
        pygame.draw.circle(screen, GREEN, EXIT_POS, EXIT_RADIUS)
        
        for obstacle in obstacles:
            obstacle.draw()
        

        for color, trails in Ball.all_trails.items():
            for trail in trails:
                if len(trail) > 1:
                    pygame.draw.lines(screen, TRAIL_COLORS[color], False, trail, 1)
        
        for color, paths in Ball.successful_paths.items():
            for path in paths:
                if len(path) > 1:
                    pygame.draw.lines(screen, GREEN, False, path, 1)
        
        for ball in balls:
            ball.draw()
        
        hunter.draw()
        
        draw_stats(balls, hunter, speed)
        
        if game_over:
            victory_text = font.render("FUCK YEAH!", True, BLACK)
            text_rect = victory_text.get_rect(center=(PLAY_AREA[0]//2, HEIGHT//2))
            screen.blit(victory_text, text_rect)
            
            restart_text = font.render("НАЖМИ R", True, BLACK)
            restart_rect = restart_text.get_rect(center=(PLAY_AREA[0]//2, HEIGHT//2 + 50))
            screen.blit(restart_text, restart_rect)
        
        pygame.display.flip()
        clock.tick(60)
    
    save_all_q_tables()
    pygame.quit()

if __name__ == "__main__":
    main()
