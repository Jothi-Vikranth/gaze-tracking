from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import threading
from time import sleep
from pydantic import BaseModel
from components import caliberate, speak, Camera, gaze_tracking, display_message


app = FastAPI()
running_flag = threading.Event()


# Define CORS settings
origins = [
    # Allow your frontend's origin (React app on localhost:3000)
    "http://localhost:3000",
    # You can add more origins if necessary
]


# Add CORSMiddleware to handle preflight requests (OPTIONS) and allow POST
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow the specified origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Allow these methods
    allow_headers=["Content-Type", "Authorization"],  # Allow these headers
)


def continuously_running_function():

    camera_instance = Camera()
    camera_instance.set_camera()

    MOVE = 30

    while True:
        while running_flag.is_set():  # Keep running while the flag is set
            gaze_tracking(camera_instance, MOVE)
            sleep(0.1)
            # time.sleep(1)  # Simulate work being done
    print("Function stopped.")


class MessageItem(BaseModel):
    message: str


@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}


@app.post("/caliberate")
def trigger_caliberate():
    caliberate()
    return {"message": "Success triggering caliberation"}


@app.post("/showMessage")
def show_message(message_item: MessageItem):

    message = message_item.message
    speak(message)
    display_message(message)
    return {"message": "Success showing message"}


@app.post("/start_gaze_tracking")
def start_function():

    if not running_flag.is_set():  # Prevent starting if it's already running

        # Start the function by setting the flag
        running_flag.set()
        threading.Thread(target=continuously_running_function,
                         daemon=True).start()
        print("Started gaze tracking.")
    else:
        return {"message": "Continuous function is already running."}


@app.post("/stop_gaze_tracking")
def stop_function():
    # Start the function by setting the flag
    if running_flag.is_set():
        print("Stopped gaze tracking.")
        running_flag.clear()
        return {"message": "Stopped continuous function."}
    else:
        return {"message": "Continuous function is not running."}
