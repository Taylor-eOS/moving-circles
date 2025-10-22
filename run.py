import tkinter as tk
import random
import math

GRID_SIZE = 20
CELL_SIZE = 30
UPDATE_INTERVAL = 200
SENSOR_RANGE = 3
LEARNING_RATE = 0.1

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-10, min(10, x))))

def clamp(v, a, b):
    return a if v < a else b if v > b else v

class EdgeDetector:
    def __init__(self):
        self.w_front = 0.0
        self.w_left = 0.0
        self.w_right = 0.0
        self.b = 0.0
        self.history = []

    def predict(self, front, left, right):
        score = self.b + front * self.w_front + left * self.w_left + right * self.w_right
        return sigmoid(score)

    def update(self, front, left, right, actual_danger):
        prediction = self.predict(front, left, right)
        error = prediction - actual_danger
        self.w_front -= LEARNING_RATE * error * front
        self.w_left -= LEARNING_RATE * error * left
        self.w_right -= LEARNING_RATE * error * right
        self.b -= LEARNING_RATE * error
        self.w_front = clamp(self.w_front, -5.0, 5.0)
        self.w_left = clamp(self.w_left, -5.0, 5.0)
        self.w_right = clamp(self.w_right, -5.0, 5.0)
        self.b = clamp(self.b, -5.0, 5.0)

    def record_accuracy(self, predicted, actual):
        correct = (predicted > 0.5) == (actual > 0.5)
        self.history.append(1.0 if correct else 0.0)
        if len(self.history) > 50:
            self.history.pop(0)

    def accuracy(self):
        if not self.history:
            return 0.0
        return sum(self.history) / len(self.history)

class RobotSim:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ML Edge Detection Robot")
        self.canvas = tk.Canvas(self.root, width=GRID_SIZE*CELL_SIZE, height=GRID_SIZE*CELL_SIZE, bg="white")
        self.canvas.pack()
        self.obstacles = set()
        self.generate_desk()
        self.x = GRID_SIZE // 2
        self.y = GRID_SIZE // 2
        self.heading = 0
        self.detector = EdgeDetector()
        self.circle = self.canvas.create_oval(0,0,0,0,fill="blue")
        self.step_count = 0
        self.draw_grid()
        self.update_robot()
        self.root.mainloop()

    def generate_desk(self):
        for i in range(GRID_SIZE):
            self.obstacles.add((i, 0))
            self.obstacles.add((i, GRID_SIZE-1))
            self.obstacles.add((0, i))
            self.obstacles.add((GRID_SIZE-1, i))
        for _ in range(15):
            ox = random.randint(3, GRID_SIZE-4)
            oy = random.randint(3, GRID_SIZE-4)
            self.obstacles.add((ox, oy))

    def draw_grid(self):
        for ox, oy in self.obstacles:
            self.canvas.create_rectangle(ox*CELL_SIZE, oy*CELL_SIZE, (ox+1)*CELL_SIZE, (oy+1)*CELL_SIZE, fill="gray")

    def sense_distance(self, dx, dy):
        for dist in range(1, SENSOR_RANGE + 1):
            check_x = self.x + dx * dist
            check_y = self.y + dy * dist
            if (check_x, check_y) in self.obstacles:
                return (SENSOR_RANGE - dist + 1) / SENSOR_RANGE
        return 0.0

    def get_sensor_readings(self):
        dirs = [(0, -1), (-1, 0), (1, 0), (0, 1)]
        dx, dy = dirs[self.heading]
        front = self.sense_distance(dx, dy)
        left_heading = (self.heading + 1) % 4
        lx, ly = dirs[left_heading]
        left = self.sense_distance(lx, ly)
        right_heading = (self.heading - 1) % 4
        rx, ry = dirs[right_heading]
        right = self.sense_distance(rx, ry)
        return front, left, right

    def is_dangerous_move(self, dx, dy):
        next_x = self.x + dx
        next_y = self.y + dy
        if (next_x, next_y) in self.obstacles:
            return True
        for check_dist in range(1, 3):
            ahead_x = next_x + dx * check_dist
            ahead_y = next_y + dy * check_dist
            if (ahead_x, ahead_y) in self.obstacles:
                return True
        return False

    def update_robot(self):
        self.step_count += 1
        front, left, right = self.get_sensor_readings()
        danger_score = self.detector.predict(front, left, right)
        dirs = [(0, -1), (-1, 0), (1, 0), (0, 1)]
        dx, dy = dirs[self.heading]
        actual_danger = 1.0 if self.is_dangerous_move(dx, dy) else 0.0
        self.detector.update(front, left, right, actual_danger)
        self.detector.record_accuracy(danger_score, actual_danger)
        if danger_score > 0.5:
            self.heading = (self.heading + random.choice([-1, 1])) % 4
        else:
            if random.random() < 0.15:
                self.heading = (self.heading + random.choice([-1, 1])) % 4
            else:
                new_x = self.x + dx
                new_y = self.y + dy
                if (new_x, new_y) not in self.obstacles:
                    self.x = new_x
                    self.y = new_y
                else:
                    self.heading = (self.heading + random.choice([-1, 1])) % 4
        acc = self.detector.accuracy()
        dir_names = ["UP", "LEFT", "RIGHT", "DOWN"]
        print(f"Step {self.step_count}: pos=({self.x},{self.y}), heading={dir_names[self.heading]}")
        print(f"  Sensors: front={front:.2f}, left={left:.2f}, right={right:.2f}")
        print(f"  Danger prediction={danger_score:.2f}, actual={actual_danger:.0f}, accuracy={acc:.2f}")
        print(f"  Weights: front={self.detector.w_front:.2f}, left={self.detector.w_left:.2f}, right={self.detector.w_right:.2f}, bias={self.detector.b:.2f}")
        print("---")
        cx = self.x * CELL_SIZE + CELL_SIZE // 2
        cy = self.y * CELL_SIZE + CELL_SIZE // 2
        r = CELL_SIZE // 3
        self.canvas.coords(self.circle, cx-r, cy-r, cx+r, cy+r)
        self.root.after(UPDATE_INTERVAL, self.update_robot)

RobotSim()
