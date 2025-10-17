import tkinter as tk
import random

GRID_SIZE = 10
CELL_SIZE = 50
NUM_CIRCLES = 3
UPDATE_INTERVAL = 1000

class MovingCircles:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Random Moving Circles")
        self.canvas = tk.Canvas(self.root, width=GRID_SIZE*CELL_SIZE, height=GRID_SIZE*CELL_SIZE, bg="white")
        self.canvas.pack()
        self.positions = [(random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)) for _ in range(NUM_CIRCLES)]
        self.circles = [self.canvas.create_oval(0,0,0,0,fill=color) for color in ("red","green","blue")]
        self.draw_grid()
        self.update_positions()
        self.root.mainloop()

    def draw_grid(self):
        for i in range(GRID_SIZE+1):
            self.canvas.create_line(i*CELL_SIZE,0,i*CELL_SIZE,GRID_SIZE*CELL_SIZE)
            self.canvas.create_line(0,i*CELL_SIZE,GRID_SIZE*CELL_SIZE,i*CELL_SIZE)

    def move_circle(self, x, y):
        directions = [(0,1),(0,-1),(1,0),(-1,0)]
        valid = []
        for dx, dy in directions:
            nx = x + dx
            ny = y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                valid.append((nx, ny))
        if not valid:
            return x, y
        return random.choice(valid)

    def update_positions(self):
        for i in range(NUM_CIRCLES):
            self.positions[i] = self.move_circle(*self.positions[i])
            x, y = self.positions[i]
            self.canvas.coords(self.circles[i],
                               x*CELL_SIZE+5, y*CELL_SIZE+5,
                               (x+1)*CELL_SIZE-5, (y+1)*CELL_SIZE-5)
        self.root.after(UPDATE_INTERVAL, self.update_positions)

MovingCircles()

