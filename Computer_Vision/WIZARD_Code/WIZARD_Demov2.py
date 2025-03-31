#!/usr/bin/env python
# coding: utf-8

# In[4]:

import os  # Import the os module 
import tensorflow as tf  # Import TensorFlow for ML operations

num_threads = os.cpu_count()  # Get the number of CPU threads available
tf.config.threading.set_intra_op_parallelism_threads(num_threads)  # Set TF's intra-op parallelism threads
tf.config.threading.set_inter_op_parallelism_threads(num_threads)  # Set TF's inter-op parallelism threads

import sys  # Import the sys module 
sys.path.append('/usr/lib/python3/dist-packages')  # Add system packages location to the module search path
import Jetson.GPIO as GPIO  # Import Jetson.GPIO to control GPIO pins 
import smbus  # Import smbus for I2C communication

import threading  # Import threading for running parallel threads
import cv2  # Import OpenCV for image & video processing
import time  # Import time for timing operations & delays
import numpy as np  # Import NumPy for numerical operations & array handling
from datetime import datetime  # Import datetime for handling date & time
from tensorflow.keras.models import load_model  # Import load_model to load model
import queue  # Import queue 

# -------------------- HIGH VOLTAGE ZAPPER CODE -----------------
bus = smbus.SMBus(1)  # Initialize I2C bus number 1 for communication with the high voltage system
board = 0x20  # Set the I2C address of the high voltage system board

# CONSTANT DEFINITIONS
IN_REG = 0x00  # Define the input register address
OUT_REG = 0x01  # Define the output register address
CONFIG_REG = 0x03  # Define the configuration register address
RESETB = 0x01  # Define RESETB: 0 means 555 is in reset, 1 means 555 is not in reset (internally pulled up)
ZAP_OFF = 0x02  # Define ZAP_OFF: 0 means zap on, 1 means zap off
CURRENT_MEASURE = 0x04  # Define CURRENT_MEASURE: 0 sends current straight to ground; 1 sends it through a comparator
CURRENT_HIGH = 0x08  # Define CURRENT_HIGH for detecting high current
SHUT_OFF = 0x10  # Define SHUT_OFF: 0 means electrode connected, 1 means electrode disconnected
CONFIG_MASK = 0xE8  # Define CONFIG_MASK used for board configuration

def initialize():
    """Set configuration of I/Os and default outputs."""
    try:
        bus.write_byte_data(board, CONFIG_REG, CONFIG_MASK)  # Write the configuration mask to the configuration register
        bus.write_byte_data(board, OUT_REG, ZAP_OFF | RESETB)  # Set default outputs: zap off and 555 not in reset
    except Exception as e:
        print("Error in initialize:", e)  # Print error if initialization fails

def EMO():
    """Emergency Off: Physically disconnect electrode from high voltage."""
    try:
        bus.write_byte_data(board, OUT_REG, SHUT_OFF)  # Write the shut off command to the output register
    except Exception as e:
        print("Error in EMO:", e)  # Print error if emergency off fails

def start_charge(measure):
    """Connect generator to 12V, activate 555 oscillator.
    Args: measure (bool): if True, measure current; if False, connect straight to ground."""
    try:
        # Write to the output register with RESETB and conditionally CURRENT_MEASURE based on the 'measure' flag
        bus.write_byte_data(board, OUT_REG, RESETB | (CURRENT_MEASURE if measure else 0))
    except Exception as e:
        print("Error in start_charge:", e)  # Print error if starting charge fails

def end_charge():
    """Disconnect generator from 12V."""
    try:
        bus.write_byte_data(board, OUT_REG, ZAP_OFF)  # Write the zap off command to disconnect the generator
    except Exception as e:
        print("Error in end_charge:", e)  # Print error if ending charge fails

def toggle_current_measure():
    """
    Toggle between lower electrode connecting to high current detector and ground.
    """
    try:
        current_setting = bus.read_byte_data(board, IN_REG)  # Read current settings from the input register
        bus.write_byte_data(board, OUT_REG, current_setting ^ CURRENT_MEASURE)  # Toggle CURRENT_MEASURE bit and write back
    except Exception as e:
        print("Error in toggle_current_measure:", e)  # Print error if toggling fails

def read_current_high():
    """Read whether high current is detected."""
    try:
        # Return the bitwise AND of the input register value with CURRENT_HIGH
        return bus.read_byte_data(board, IN_REG) & CURRENT_HIGH
    except Exception as e:
        print("Error in read_current_high:", e)  # Print error if reading fails
        return None  # Return None in case of error

def trigger_zap():
    """
    Safely trigger a zap by starting and then ending the charge.
    This routine is wrapped in a try/except block so that any errors (for instance,
    if a BIT fault causes the board to reset) do not crash the system.
    """
    try:
        print("Triggering zap...")  # Notify that zap is being triggered
        start_charge(False)  # Start charge with measure set to False to trigger zap
        time.sleep(0.5)  # Wait set amount of seconds for the zap pulse duration
        end_charge()  # End charge to disconnect the generator
        print("Zap triggered successfully.\n")  # Notify successful zap
    except Exception as e:
        print("Error during zap trigger:", e)  # Print error if zap triggering fails
# -------------------- END ZAPPER CODE --------------------

# -------------------- WEED DETECTION CODE --------------------
# Global constants and variables for detection
BIT_LED = 7  # Define the GPIO pin number for the red LED (BIT LED)

model_path = 'WIZARDv15.h5'  # Set path for the ML model file
tracker = load_model(model_path)  # Load the trained model from the specified file
CROP_COLOR = (0, 255, 0)  # Define the BGR color for crops (green)
WEED_COLOR = (0, 0, 255)  # Define the BGR color for weeds (red)
score_threshold = 0.85  # Set the confidence score threshold to 0.85

# Global variables for inter-thread communication
latest_predictions = None  # Variable to store the latest model prediction
prediction_lock = threading.Lock()  # Create a lock for synchronizing prediction updates
running = True  # Flag to signal threads to continue running

# Global variables for weed detection stability
prev_weed_bbox = None  # Store the previous weed bounding box (start_x, start_y, end_x, end_y)
stable_detection_printed = False  # Flag to ensure stable detection is printed only once per cycle
stable_start_time = None  # Timestamp when stable detection condition was first met
stable_detection_count = 0  # Counts consecutive stable detections

# Create a queue to always hold the latest frame
frame_queue = queue.Queue(maxsize=1)

def init_gpio():
    """Initialize GPIO settings for BIT LED."""
    GPIO.setmode(GPIO.BOARD)  # Set the GPIO pin numbering mode to BOARD
    GPIO.setup(BIT_LED, GPIO.OUT)  # Set BIT_LED pin as an output
    GPIO.output(BIT_LED, 0)  # Turn off BIT_LED initially

def preprocess_frame(frame, input_size=(224, 224)):
    """Preprocess the frame for model input."""
    frame_resized = cv2.resize(frame, input_size)  # Resize frame to input dimensions
    frame_normalized = frame_resized / 255.0  # Normalize pixel values to [0, 1]
    return np.expand_dims(frame_normalized, axis=0)  # Add batch dimension and return frame

def parallel_predict(frame, model, input_size=(224, 224)):
    """Perform prediction on the frame."""
    preprocessed_frame = preprocess_frame(frame, input_size)  # Preprocess the frame
    prediction = model.predict(preprocessed_frame)  # Get model prediction
    return prediction  # Return prediction

def draw_bounding_boxes(frame, yhat, previous_predictions=None, iou_threshold=0.5, score_threshold=0.85):
    """Draw bounding boxes on the frame using model predictions."""
    global prev_weed_bbox, stable_detection_printed, stable_start_time, stable_detection_count  # Declare globals for stability tracking and counter

    num_predictions = 3  # Only consider the first 3 predictions
    class_preds, bbox_preds = yhat  # Unpack predictions into class and bounding box arrays from yhat
    class_probs = class_preds[0][:num_predictions]  # Get class probabilities for the first 3 predictions
    bbox_coords = bbox_preds[0][:num_predictions]  # Get bounding box coordinates for the first 3 predictions

    weed_boxes = []  # Initialize an empty list to hold weed bounding boxes
    weed_scores = []  # Initialize an empty list to hold weed detection scores
    weed_classes = []  # Initialize an empty list to hold weed class labels
    crop_detected = False  # Flag to indicate if a crop or background is detected

    for box_idx in range(num_predictions):  # Loop over each of the first 3 predictions
        prob_crop, prob_weed, prob_background = class_probs[box_idx]  # Unpack probabilities for crop, weed, and background
        if prob_crop > prob_weed and prob_crop > score_threshold:  # If crop probability is dominant and above threshold
            crop_detected = True  # Mark that a crop is detected
            # Crop detected; do not display crop classification
        elif prob_weed > prob_crop and prob_weed > score_threshold:  # If weed probability is dominant and above threshold
            weed_boxes.append(bbox_coords[box_idx])  # Add current bounding box to weed_boxes list
            weed_scores.append(prob_weed)  # Add weed detection score to weed_scores list
            weed_classes.append("Weed")  # Add label "Weed" to weed_classes list

    if crop_detected:  # If a crop or background is detected in any prediction
        prev_weed_bbox = None  # Clear previous weed bounding box
        stable_detection_printed = False  # Reset flag for stable detection printed
        stable_start_time = None  # Reset the stability detection start time
        stable_detection_count = 0  # Reset the stable detection counter

    if len(weed_boxes) > 0:  # If there are any weed detections
        weed_boxes = np.array(weed_boxes)  # Convert weed_boxes list to a NumPy array
        weed_scores = np.array(weed_scores)  # Convert weed_scores list to a NumPy array
        weed_classes = np.array(weed_classes)  # Convert weed_classes list to a NumPy array
        
        # Perform non-max suppression on the weed boxes to eliminate redundant detections
        selected_indices = tf.image.non_max_suppression(weed_boxes, weed_scores, max_output_size=5, iou_threshold=iou_threshold)  # Apply NMS to weed boxes
        selected_boxes = tf.gather(weed_boxes, selected_indices).numpy()  # Gather selected boxes and convert to NumPy array
        selected_scores = tf.gather(weed_scores, selected_indices).numpy()  # Gather corresponding scores
        selected_classes = weed_classes[selected_indices.numpy()]  # Get corresponding class labels for selected boxes

        for i in range(len(selected_boxes)):  # Loop over each selected detection
            box = selected_boxes[i]  # Get bounding box for current detection
            class_name = selected_classes[i]  # Get class label 
            score = selected_scores[i]  # Get detection score for current detection
            color = WEED_COLOR  # Set color to red
            frame_height, frame_width = frame.shape[:2]  # Get frame dimensions (height and width)
            start_point = tuple(np.multiply(box[:2], [frame_width, frame_height]).astype(int))  # Top-left corner in pixel coordinates
            end_point = tuple(np.multiply(box[2:], [frame_width, frame_height]).astype(int))  # Bottom-right corner in pixel coordinates
            original_width = end_point[0] - start_point[0]  # Calculate the original width of the bounding box
            if original_width > 40:  # If the box's width is more than 40 pixels
                new_start_point = (start_point[0] + 20, start_point[1])  # Reduce the left side by 20 pixels
                new_end_point = (end_point[0] - 20, end_point[1])  # Reduce the right side by 20 pixels
            else:  # If the box is not wide enough
                new_start_point, new_end_point = start_point, end_point  # Use original coordinates without modification
            cv2.rectangle(frame, new_start_point, new_end_point, color, 2)  # Draw the adjusted bounding rectangle on the frame
            cv2.putText(frame, f'{class_name}: {score:.2f}', new_start_point,  # Draw the label and score text at the new start point
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)  # Use specified font, size, color, thickness, and line type
            center_point = ((new_start_point[0] + new_end_point[0]) // 2, (new_start_point[1] + new_end_point[1]) // 2)  # Calculate the center of the adjusted box
            cv2.circle(frame, center_point, 5, color, -1)  # Draw a filled circle at the center point

        rep_idx = np.argmax(selected_scores)  # Identify the index of the detection with the highest score
        rep_box = selected_boxes[rep_idx]  # Get the representative bounding box corresponding to the highest score
        frame_height, frame_width = frame.shape[:2]  # Retrieve frame dimensions again
        rep_start_point = tuple(np.multiply(rep_box[:2], [frame_width, frame_height]).astype(int))  # Top-left corner for the representative box in pixels
        rep_end_point = tuple(np.multiply(rep_box[2:], [frame_width, frame_height]).astype(int))  # Bottom-right corner for the representative box in pixels
        rep_width = rep_end_point[0] - rep_start_point[0]  # Calculate the width of the representative bounding box
        if rep_width > 40:  # If the representative box is wider than 40 pixels
            rep_start_point = (rep_start_point[0] + 20, rep_start_point[1])  # Adjust the left coordinate by 20 pixels
            rep_end_point = (rep_end_point[0] - 20, rep_end_point[1])  # Adjust the right coordinate by 20 pixels
        rep_center_x = (rep_start_point[0] + rep_end_point[0]) // 2  # X-coordinate of the representative box's center
        rep_center_y = (rep_start_point[1] + rep_end_point[1]) // 2  # Y-coordinate of the representative box's center

        if prev_weed_bbox is not None:  # If a previous weed bounding box exists
            prev_start_x, prev_start_y, prev_end_x, prev_end_y = prev_weed_bbox  # Unpack previous bounding box coordinates
            prev_center_x = (prev_start_x + prev_end_x) // 2  # X-coordinate of the previous box's center
            prev_center_y = (prev_start_y + prev_end_y) // 2  # Y-coordinate of the previous box's center
            movement_threshold = 20  # Define the pixel threshold for stable movement
            if (abs(rep_center_x - prev_center_x) < movement_threshold and
                abs(rep_center_y - prev_center_y) < movement_threshold):  # If current detection is within the threshold of previous detection
                # If a zap has already been triggered and the detection remains stable then do nothing.
                if stable_detection_printed:
                    pass  # Maintain the current state; do not restart the stability sequence.
                else:
                    if stable_start_time is None:  # If the stability timer has not been started
                        stable_start_time = time.time()  # Start the stability timer by recording the current time
                    else:
                        if time.time() - stable_start_time >= 3:  # If the detection has remained stable for at least 3 seconds
                            if stable_detection_count == 0:  # First stable detection cycle
                                stable_detection_count = 1  # Increment the stable detection counter
                                print("\nFirst weed detection confirmed:", (rep_center_x, rep_center_y), "\n")
                                stable_start_time = time.time()  # Reset timer for the next stability check
                            else:  # Second consecutive stable detection cycle
                                print("\nStable weed detection center:", (rep_center_x, rep_center_y), "\n")
                                trigger_zap()  # Trigger the zap action now that location is confirmed twice
                                stable_detection_printed = True  # Mark that a zap has been triggered
                                # Do not reset stable_start_time or stable_detection_count here if the center remains stable;
                                # the state is maintained until the detection center changes.
            else:  # If the detection has moved outside the threshold
                stable_start_time = None  # Reset the stability timer
                stable_detection_printed = False  # Reset the flag indicating a zap was triggered
                stable_detection_count = 0  # Reset the stable detection counter
        else:  # If no previous weed bounding box exists
            stable_start_time = time.time()  # Start the stability timer for the first weed detection
            stable_detection_printed = False  # Ensure the zap-trigger flag is reset
            stable_detection_count = 0  # Initialize the stable detection counter

        # Update the global previous weed bounding box with the adjusted representative box coordinates
        prev_weed_bbox = (rep_start_point[0], rep_start_point[1], rep_end_point[0], rep_end_point[1])
    else:  # If there are no weed detections
        prev_weed_bbox = None  # Clear the previous weed bounding box
        stable_detection_printed = False  # Reset the zap-trigger flag
        stable_start_time = None  # Reset the stability timer
        stable_detection_count = 0  # Reset the stable detection counter

    if previous_predictions:  # If previous predictions are provided
        for pred in previous_predictions:  # Loop over each previous prediction
            if isinstance(pred[0], tuple):  # Check if the prediction tuple represents a bounding box
                box, class_name, score, color = pred  # Unpack the prediction tuple into box, class name, score, and color
                frame_height, frame_width = frame.shape[:2]  # Get frame dimensions
                start_point = tuple(np.multiply(box[:2], [frame_width, frame_height]).astype(int))  # Top-left corner in pixel coordinates
                end_point = tuple(np.multiply(box[2:], [frame_width, frame_height]).astype(int))  # Bottom-right corner in pixel coordinates
                original_width = end_point[0] - start_point[0]  # Calculate the original width of the bounding box
                if original_width > 40:  # If the box is wider than 40 pixels
                    new_start_point = (start_point[0] + 20, start_point[1])  # Adjust the left coordinate by 20 pixels
                    new_end_point = (end_point[0] - 20, end_point[1])  # Adjust the right coordinate by 20 pixels
                else:  # If the box is not wide enough
                    new_start_point, new_end_point = start_point, end_point  # Use the original coordinates
                cv2.rectangle(frame, new_start_point, new_end_point, color, 2)  # Draw the adjusted rectangle for the previous prediction
                cv2.putText(frame, f'{class_name}: {score:.2f}', new_start_point,  # Draw the label and score text for the previous prediction
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)  # Use specified font, size, color, thickness, and line type

    return frame  # Return the annotated frame with all drawn bounding boxes and labels

def prediction_thread(model):
    global latest_predictions, running  # Use globals for latest predictions and running flag
    while running:  # Continue looping while running is True
        try:
            frame_copy = frame_queue.get(timeout=0.05)  # Get a frame from the queue with a timeout
        except queue.Empty:
            continue  # If queue is empty, continue the loop
        predictions = parallel_predict(frame_copy, model)  # Compute predictions on the frame
        with prediction_lock:
            latest_predictions = predictions  # Update global predictions with thread-safety

def real_time_detection(model):
    """Run real-time detection while predictions run in parallel.
       Returns True if detection completes normally, False if a BIT fault occurs."""
    global running  # Use the global running flag
    init_gpio()  # Reinitialize GPIO settings for BIT LED
    user_quit = False  # Flag to indicate user-initiated exit (unused further in code)

    last_fps_update_time = time.time()  # Initialize last FPS update time
    cached_fps_text = "FPS: 0.00"  # Initialize cached FPS display text

    gst_pipeline = (  # Define GStreamer pipeline for camera capture
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=1280, height=720, framerate=60/1 ! "
        "nvvidconv flip-method=2 ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! appsink"
    )
    
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)  # Open video capture with the specified pipeline
    timeout = 5.0  # Set timeout for camera initialization
    start_time = time.time()  # Record start time for camera initialization
    while not cap.isOpened() and (time.time() - start_time) < timeout:  # Wait until the camera opens or timeout is reached
        time.sleep(0.1)  # Sleep briefly between attempts
    if not cap.isOpened():  # If camera failed to open
        print("Failed to open camera with GStreamer pipeline. Triggering BIT early.")  # Print error message
        GPIO.output(BIT_LED, 1)  # Turn on BIT LED to indicate camera open failure
        time.sleep(5)  # LED on for 5 seconds
        GPIO.output(BIT_LED, 0)  # Turn off BIT LED
        return False  # Return False indicating a BIT fault
        
    prev_time = time.time()  # Record previous time for FPS calculation
    fps = 0  # Initialize FPS variable
    alpha = 0.1  # Set smoothing factor for FPS calculation

    now = datetime.now().strftime('%Y%m%d_%H%M%S')  # Get current date and time as string
    output_folder = '/media/nvidia/ESD-USB/WIZARD/Videos'  # Define output folder for video saving
    os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist
    output_filename = f'{output_folder}/{model_path.split(".")[0]}_{now}.mp4'  # Create output filename using model name and current time
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define video codec (mp4v)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get width of video frames from capture device
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Get height of video frames from capture device
    out = cv2.VideoWriter(output_filename, fourcc, 10.0, (frame_width, frame_height))  # Initialize video writer with output filename and frame dimensions
    print(f"Saving video to: {output_filename}")  # Print the path to the saved video file

    pred_thread = threading.Thread(target=prediction_thread, args=(model,), daemon=True)  # Create a daemon thread for predictions
    pred_thread.start()  # Start the prediction thread

    while True:  # Main loop for real-time detection
        ret, frame = cap.read()  # Read a frame from the camera
        if not ret:  # If frame capture fails
            print("Error: Failed to capture image.")  # Print error message for capture failure
            GPIO.output(BIT_LED, 1)  # Turn on BIT LED to indicate capture failure
            time.sleep(5)  # LED on for 5 seconds
            GPIO.output(BIT_LED, 0)  # Turn off BIT LED
            running = False  # Set running flag to False to signal prediction thread to stop
            pred_thread.join()  # Wait for prediction thread to finish
            cap.release()  # Release the camera resource
            out.release()  # Release the video writer resource
            cv2.destroyAllWindows()  # Close all OpenCV windows
            return False  # Return False indicating a BIT fault
            
        GPIO.output(BIT_LED, 0)  # Ensure BIT LED is turned off during normal operation

        if frame_queue.full():  # Check if frame queue is full
            try:
                frame_queue.get_nowait()  # Remove the oldest frame if necessary
            except queue.Empty:
                pass  # If queue is empty, do nothing
        frame_queue.put(frame.copy())  # Put a copy of the current frame into the frame queue

        curr_time = time.time()  # Get current time for FPS calculation
        instant_fps = 1 / (curr_time - prev_time)  # Calculate instantaneous FPS
        fps = (alpha * instant_fps) + (1 - alpha) * fps  # Apply exponential smoothing to FPS
        prev_time = curr_time  # Update previous time for next calculation

        if curr_time - last_fps_update_time >= 2.0:  # If more than 2 seconds have passed since last FPS update
            cached_fps_text = f"FPS: {fps:.2f}"  # Update the cached FPS text
            last_fps_update_time = curr_time  # Reset last FPS update time

        text_size, _ = cv2.getTextSize(cached_fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)  # Get text size for FPS display
        text_x = frame.shape[1] - text_size[0] - 10  # Calculate x-coordinate for FPS text placement
        text_y = frame.shape[0] - 10  # Calculate y-coordinate for FPS text placement
        cv2.putText(frame, cached_fps_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)  # Draw FPS text on the frame

        with prediction_lock:
            current_predictions = latest_predictions  # Retrieve the latest predictions in a thread-safe manner

        if current_predictions is not None:  # If there are predictions available
            output_frame = draw_bounding_boxes(frame, current_predictions)  # Draw bounding boxes on the frame
        else:
            output_frame = frame  # Otherwise, use the original frame

        out.write(output_frame)  # Write the processed frame to the video file
        cv2.imshow('Real-Time Detection', output_frame)  # Display the processed frame
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Check if the user pressed 'q' to exit
            print("\nClosing system as per user request. Turning off...\n")  # Print exit message
            break  # Exit the main loop

    running = False  # Signal the prediction thread to stop
    pred_thread.join()  # Wait for the prediction thread to finish
    cap.release()  # Release the camera resource
    out.release()  # Release the video writer resource
    cv2.destroyAllWindows()  # Close all OpenCV windows
    
    return True  # Return True to indicate normal exit

def main():
    """Main loop that ruins model and resets the system upon BIT faults."""
    global running  # Use the global running flag
    initialize()  # Initialize the high voltage zapper board
    retry_count = 0  # Initialize a counter for BIT faults
    while True:  # Main operational loop
        running = True  # Reset the running flag for a new detection session
        result = real_time_detection(tracker)  # Run real-time detection using the loaded model
        if result is False:  # If a BIT fault occurred during detection
            retry_count += 1  # Increment the retry counter
            if retry_count == 1:  # On the first fault occurrence
                print("\nBIT fault has occurred. Resetting system...\n")  # Notify that a BIT fault occurred
                time.sleep(5)  # Wait for 5 seconds before retrying
            else:
                print("\nA BIT fault has occurred again. Would you like to exit? (yes/no): ", end='', flush=True)  # Prompt user on repeated faults
                user_input = input()  # Get user input
                if user_input.lower() in ['yes', 'y']:  # If user wants to exit
                    print("\nExiting as per user request.")  # Print exit message
                    break  # Exit the main loop
                else:
                    print("\nResetting system...\n")  # Otherwise, notify system reset
                    time.sleep(5)  # Wait for 5 seconds before retrying
        else:
            break  # If detection completed normally, exit the loop

if __name__ == '__main__':
    main()  # Run the main function when the script is executed directly

# In[ ]: