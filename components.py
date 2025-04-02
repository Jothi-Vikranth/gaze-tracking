import os
import json
import tkinter as tk
from time import sleep
import threading
from datetime import datetime
import pyautogui
from utils import Camera, Speak, get_data, collect_landmark_data_2,  X_Dir, Y_Dir, get_landmark_points, evaluate_model
from RPLCD.i2c import CharLCD
from time import sleep
import time
from datetime import datetime


data_collection_fn = collect_landmark_data_2


class MovingCircleApp:

    margin = 30
    circle_animate_timing = 1.5
    min_radius = 20
    max_radius = 50

    def __init__(self):
        # Initialize the main window
        self.root = tk.Tk()
        self.root.title("Gaze-tracking Caliberation")

        # Get screen dimensions and set the window to full screen
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{self.screen_width}x{self.screen_height}")
        self.root.attributes("-fullscreen", True)

        # Create a canvas that fills the screen
        self.canvas = tk.Canvas(
            self.root, width=self.screen_width, height=self.screen_height, bg="white")
        self.canvas.pack(fill="both", expand=True)

        # Bind Escape key to exit full-screen mode
        self.root.bind("<Escape>", lambda event: self.root.attributes(
            "-fullscreen", False))

        # Create the circle attribute
        self.radius = self.max_radius
        self.curr_radius = self.max_radius
        self.circle = self.canvas.create_oval(
            10 - self.radius, 10 - self.radius,
            10 + self.radius, 10 + self.radius,
            fill="blue"
        )
        # self.root.withdraw()

    def __get_canvas_width(self):
        return self.canvas.winfo_width(), self.canvas.winfo_height()

    def move_circle(self, inp_x: X_Dir, inp_y: Y_Dir):
        """Move the circle to a random position on the canvas."""
        canvas_width, canvas_height = self.__get_canvas_width()

        # Generate random coordinates for the circle
        if inp_x == X_Dir.LEFT:
            x = self.radius + self.margin
        elif inp_x == X_Dir.H_CENTER:
            x = int(canvas_width/2)
        elif inp_x == X_Dir.RIGHT:
            x = int(canvas_width - self.radius - self.margin)
        else:
            raise ValueError("Unknown X_dir value received")

        if inp_y == Y_Dir.TOP:
            y = self.radius + self.margin
        elif inp_y == Y_Dir.V_CENTER:
            y = int(canvas_height/2)
        elif inp_y == Y_Dir.BOTTOM:
            y = int(canvas_height - self.radius - self.margin)
        else:
            raise ValueError("Unknown X_dir value received")

        # Update the position of the circle
        self.canvas.coords(self.circle, x - self.radius, y -
                           self.radius, x + self.radius, y + self.radius)

    def animate_radius(self, animation_time: float):
        """Animate the circle by continuously changing its radius."""
        # Increase or decrease the radius
        FPS = 100
        _dir = -1

        sleep_time = 1/FPS
        inc = (self.max_radius - self.min_radius) / \
            (self.circle_animate_timing*FPS/2)

        self.curr_radius = self.radius

        for _ in range(int(animation_time*FPS)):
            start_time = datetime.now()
            # print(i, "   ", start_time)
            if self.curr_radius <= self.min_radius:
                _dir = 1
                self.curr_radius = self.min_radius
            elif self.curr_radius >= self.max_radius:
                _dir = -1
                self.curr_radius = self.max_radius

            self.curr_radius += _dir * inc
            # self.curr_radius = int(self.curr_radius)

            # Get the current position of the circle
            coords = self.canvas.coords(self.circle)
            x = (coords[0] + coords[2]) / 2  # Circle's center x
            y = (coords[1] + coords[3]) / 2  # Circle's center y

            # Update the circle's size
            self.canvas.coords(self.circle, x - self.curr_radius, y -
                               self.curr_radius, x + self.curr_radius, y + self.curr_radius)
            while (datetime.now() - start_time).total_seconds() < sleep_time:
                pass

        # Schedule the next animation step

    def process_at_position(self, x: X_Dir, y: Y_Dir):

        data_collection_time = 2
        self.move_circle(x, y)

        cam = Camera().get_camera()
        file_ls = []
        file_ls.append(x.name)
        file_ls.append(y.name)

        thread_gui = threading.Thread(
            target=self.animate_radius, args=(data_collection_time+0.3,))
        thread_data_collection = threading.Thread(
            target=get_data, args=(cam, collect_landmark_data_2, file_ls, data_collection_time))

        thread_gui.start()
        sleep(0.3)
        thread_data_collection.start()
        # Speak().speak("Data collecting")
        thread_gui.join()
        thread_data_collection.join()
        # Speak().speak("Completed")

    def background_task(self):
        sleep(1)
        Speak().speak("Please look into the circle for caliberation")
        sleep(4)

        self.process_at_position(X_Dir.LEFT, Y_Dir.TOP)
        self.process_at_position(X_Dir.H_CENTER, Y_Dir.TOP)
        self.process_at_position(X_Dir.RIGHT, Y_Dir.TOP)
        self.process_at_position(X_Dir.RIGHT, Y_Dir.V_CENTER)
        self.process_at_position(X_Dir.RIGHT, Y_Dir.BOTTOM)
        self.process_at_position(X_Dir.H_CENTER, Y_Dir.BOTTOM)
        self.process_at_position(X_Dir.LEFT, Y_Dir.BOTTOM)
        self.process_at_position(X_Dir.LEFT, Y_Dir.V_CENTER)
        self.process_at_position(X_Dir.H_CENTER, Y_Dir.V_CENTER)

        self.close_window()

    def run(self):
        """Start the application."""
        threading.Thread(target=self.background_task).start()
        self.root.mainloop()

    def close_window(self):
        """Close the tkinter window."""

        self.root.withdraw()
        self.root.quit()
        # self.root.after(0, self.root.quit)  # Close the window safely


def remove_files_if_exist():
    file_ls = []
    file_ls.extend([member.name for member in X_Dir])
    file_ls.extend([member.name for member in Y_Dir])
    file_ls = list(set(file_ls))

    whole_data_folder = "whole_data"

    if not os.path.exists(whole_data_folder):
        os.makedirs(whole_data_folder)

    for file in file_ls:
        file_name = f'{file}.json'
        whole_data_file_path = os.path.join(whole_data_folder, file_name)

        with open(whole_data_file_path, 'w') as file:
            json.dump([], file, indent=4)

        if os.path.exists(file_name):
            os.remove(file_name)


def caliberate():
    # remove_files_if_exist()

    camera_instance = Camera()
    camera_instance.set_camera()

    app = MovingCircleApp()
    app.run()
    app.close_window()


def speak(message):
    Speak().speak(message)


def gaze_tracking(camera_instance, MOVE):

    corner_tolerance = 10
    screen_width, screen_height = pyautogui.size()

    frame = camera_instance.get_frame()

    landmark_points = get_landmark_points(frame)
    frame_h, frame_w, _ = frame.shape

    if landmark_points:

        landmarks = landmark_points[0].landmark
        data = data_collection_fn(landmarks, frame_w, frame_h)
        horizontal, vertical = evaluate_model([data])

        current_x, current_y = pyautogui.position()

        dx = getattr(X_Dir, horizontal).value * MOVE
        dy = getattr(Y_Dir, vertical).value * MOVE

        x_pos = current_x + dx
        y_pos = current_y + dy

        if x_pos < corner_tolerance or x_pos > screen_width - corner_tolerance:
            if y_pos < corner_tolerance or y_pos > screen_height - corner_tolerance:
                return
        pyautogui.moveTo(x_pos, y_pos)


def display_message(msg):
    # Adjust I2C address if needed (use i2cdetect to check)
    lcd = CharLCD(i2c_expander='PCF8574', address=0x27,
                  port=1, cols=16, rows=2)

    lcd.clear()
    lcd.write_string(msg)
    sleep(1)
