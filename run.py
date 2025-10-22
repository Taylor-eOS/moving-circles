import tkinter as tk
import random
import math
from collections import deque

GRID_SIZE = 10
CELL_SIZE = 50
UPDATE_INTERVAL = 150
ACTION_DIM = 4
FEATURES_PER_ACTION = 1
INPUT_DIM = ACTION_DIM * FEATURES_PER_ACTION
LEARNING_RATE = 0.3

def softmax(logits):
    m = max(logits)
    exps = [math.exp(l - m) for l in logits]
    s = sum(exps)
    if s == 0:
        return [1.0 / len(logits)] * len(logits)
    return [e / s for e in exps]

def sample_index(probs):
    r = random.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if r <= cum:
            return i
    return len(probs) - 1

def clamp(v, a, b):
    return a if v < a else b if v > b else v

class Agent:
    def __init__(self):
        self.features_per_action = 1
        self.W = [[0.0 for _ in range(self.features_per_action)] for _ in range(ACTION_DIM)]
        self.b = [0.0 for _ in range(ACTION_DIM)]
        self.visited = deque()
        self.visited_set = set()
        self.last_probs = None
        self.last_input = None
        self.correct_history = deque(maxlen=50)

    def get_input(self, own):
        x, y = own
        vs = self.visited_set
        deltas = [(0,1),(0,-1),(1,0),(-1,0)]
        out = []
        for dx, dy in deltas:
            nx = x + dx
            ny = y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                idx = nx + ny * GRID_SIZE
                is_unvisited = 1.0 if idx not in vs else 0.0
            else:
                is_unvisited = 0.0
            out.append(is_unvisited)
        return out

    def add_visited(self, cell_index):
        if cell_index in self.visited_set:
            return
        self.visited.append(cell_index)
        self.visited_set.add(cell_index)
        if len(self.visited) > 50:
            old = self.visited.popleft()
            self.visited_set.discard(old)

    def record_choice(self, was_best):
        self.correct_history.append(1.0 if was_best else 0.0)

    def rolling_accuracy(self):
        if not self.correct_history:
            return 0.0
        return sum(self.correct_history) / len(self.correct_history)

    def forward(self, inp, valid_mask):
        logits = []
        for a in range(ACTION_DIM):
            base = self.b[a]
            start = a * self.features_per_action
            s = base
            for f in range(self.features_per_action):
                s += inp[start+f] * self.W[a][f]
            if not valid_mask[a]:
                s = -1e9
            logits.append(s)
        probs = softmax(logits)
        self.last_probs = probs
        self.last_input = inp
        return probs

    def update(self, chosen_action, reward):
        if self.last_input is None:
            return
        x = self.last_input
        a = chosen_action
        if x[a] > 0.5:
            delta = reward * 0.1
            self.W[a][0] += delta
            self.W[a][0] = clamp(self.W[a][0], -10.0, 10.0)

class MovingCircle:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ML moving circle")
        self.canvas = tk.Canvas(self.root, width=GRID_SIZE*CELL_SIZE, height=GRID_SIZE*CELL_SIZE, bg="white")
        self.canvas.pack()
        self.position = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
        self.agent = Agent()
        start_index = self.position[0] + self.position[1] * GRID_SIZE
        self.agent.add_visited(start_index)
        self.circle = self.canvas.create_oval(0,0,0,0,fill="blue")
        self.draw_grid()
        self.step_count = 0
        self.update_position()
        self.root.mainloop()

    def draw_grid(self):
        for i in range(GRID_SIZE+1):
            self.canvas.create_line(i*CELL_SIZE,0,i*CELL_SIZE,GRID_SIZE*CELL_SIZE)
            self.canvas.create_line(0,i*CELL_SIZE,GRID_SIZE*CELL_SIZE,i*CELL_SIZE)

    def valid_actions_mask(self, x, y):
        mask = [True, True, True, True]
        if y + 1 >= GRID_SIZE:
            mask[0] = False
        if y - 1 < 0:
            mask[1] = False
        if x + 1 >= GRID_SIZE:
            mask[2] = False
        if x - 1 < 0:
            mask[3] = False
        return mask

    def action_to_delta(self, a):
        if a == 0:
            return (0, 1)
        if a == 1:
            return (0, -1)
        if a == 2:
            return (1, 0)
        return (-1, 0)

    def count_new_adjacent(self, x, y):
        count = 0
        adjacent = [(x, y+1), (x, y-1), (x+1, y), (x-1, y)]
        for ax, ay in adjacent:
            if 0 <= ax < GRID_SIZE and 0 <= ay < GRID_SIZE:
                cell_index = ax + ay * GRID_SIZE
                if cell_index not in self.agent.visited_set:
                    count += 1
        return count

    def update_position(self):
        self.step_count += 1
        x, y = self.position
        inp = self.agent.get_input(self.position)
        mask = self.valid_actions_mask(x, y)
        probs = self.agent.forward(inp, mask)
        action = sample_index(probs)
        dx, dy = self.action_to_delta(action)
        nx = clamp(x + dx, 0, GRID_SIZE-1)
        ny = clamp(y + dy, 0, GRID_SIZE-1)
        cell_index = nx + ny * GRID_SIZE
        reward = 1.0 if cell_index not in self.agent.visited_set else -1.0
        self.agent.update(action, reward)
        self.agent.add_visited(cell_index)
        self.position = (nx, ny)
        best_action = -1
        best_new_count = -1
        for a in range(ACTION_DIM):
            if mask[a]:
                dx_test, dy_test = self.action_to_delta(a)
                test_x = clamp(x + dx_test, 0, GRID_SIZE-1)
                test_y = clamp(y + dy_test, 0, GRID_SIZE-1)
                new_count = self.count_new_adjacent(test_x, test_y)
                if new_count > best_new_count:
                    best_new_count = new_count
                    best_action = a
        chose_best = True if action == best_action else False
        self.agent.record_choice(chose_best)
        rolling = self.agent.rolling_accuracy()
        action_names = ["down", "up", "right", "left"]
        print(f"Step {self.step_count}: pos=({nx},{ny}), action={action_names[action]}, reward={reward:.1f}")
        print(f"  New adjacent: {self.count_new_adjacent(nx, ny)}, Correct: {'YES' if chose_best else 'NO'}, RollingAcc={rolling:.2f}")
        print(f"  Action probs: down={probs[0]:.2f}, up={probs[1]:.2f}, right={probs[2]:.2f}, left={probs[3]:.2f}")
        w_str = f"  Weights[unvisited]: down={self.agent.W[0][0]:.2f}, up={self.agent.W[1][0]:.2f}, right={self.agent.W[2][0]:.2f}, left={self.agent.W[3][0]:.2f}"
        print(w_str)
        print("---")
        self.canvas.coords(self.circle, nx*CELL_SIZE+5, ny*CELL_SIZE+5, (nx+1)*CELL_SIZE-5, (ny+1)*CELL_SIZE-5)
        self.root.after(UPDATE_INTERVAL, self.update_position)

MovingCircle()
