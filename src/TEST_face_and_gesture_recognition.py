import cv2
import numpy as np
import face_recognition
import pickle
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from picamera2 import Picamera2

import statistics # Used to calculate metrics like mean and standard deviation

# Test configuration 
# Set to "True" to run in test mode, "False" for normal operation
RUN_TEST_MODE = True
TEST_DURATION_SECONDS = 30 # For how many seconds the test should run


# Test data storage structures (acting as lists)
fps_readings = []
face_detected_log = []
gesture_detected_log = []
both_detected_log = []



# Load pre-trained face encodings
print("[INFO] loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]

# Initialize the MediaPipe Gesture Recognizer
base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options, running_mode=vision.RunningMode.IMAGE, num_hands = 2)
gesture_recognizer = vision.GestureRecognizer.create_from_options(options)

#  Create specialised responses to gestures with a gesture-response mapping dictionary
gesture_text = {
     "Victory": "Peace, {username}!",
     "Thumb_Up": "Keep going, {username}!",
     "Thumb_Down": "You can do better, {username}!",
     "Open_Palm": "Hello, {username} :]",
     "Pointing_Up": "Whats up, {username}?",
     "ILoveYou": "Love you too, {username} ;]",
     "Closed_Fist": "Keep it tight, {username}!",
     "None": "I got nothing, {username}", 
 }

# Initialize Raspberry Pi Camera Module 3
picam2 = Picamera2()

# preview_config = picam2.create_preview_configuration(main={"size": (640, 480)})
# The camera output format is set to XRGB8888 to avoid alpha channel issues 
preview_config = picam2.create_preview_configuration( main = { "size": (1920, 1000), "format": "XRGB8888"},
sensor = {"output_size": (2304, 1296), "bit_depth": 10}, encode = "main") 
# A 16:9 aspect ratio is chosen for the camera module
# and the raw sensor output resolution is set to 2304x1296 out of the camera sensor's 4056x3040 pixel resolution
# This is done to achieve a wider field of view for the camera module
# encode = "main" allows the H.264 encoder block to be used to speed up processing
picam2.configure(preview_config)
picam2.start()

# Initialize the variables
cv_scaler = 6 # this has to be a whole number, the resolution of the input frame is reduced by
# A factor of 1/cv-scaler, the higher cv_scaler is the faster processing becomes 
frame_count = 0
face_locations = []
face_encodings = []
face_names = []
test_start_time = time.time() # Used to determine the time this run of the test began
fps_start_time = test_start_time # Used to calculate the rolling average FPS 
fps = 0

cv2.namedWindow('Complete Recognition', cv2.WINDOW_NORMAL)
# WINDOW_NORMAL allows the displayed window to be resized with manual sizing
cv2.resizeWindow('Complete Recognition', 1280, 720) 
# Defines the size of the displayed window

if RUN_TEST_MODE:
    print(f"[INFO] Test will run for {TEST_DURATION_SECONDS} seconds.")

try:
    while True:
        frame_start_time = time.time() # Start timing this frame for detailed FPS
        elapsed_time = frame_start_time - test_start_time
        
        # Test duration check:
        if RUN_TEST_MODE and (elapsed_time > TEST_DURATION_SECONDS):
            print(f"[INFO] Test duration completed, {elapsed_time:.2f} seconds elapsed.")
            break
            
        # Capture frame from Pi Camera and convert it into a BGR format 
        # Ensuring that the captrued frame has only 3 channels (the alpha channel is trimmed and made contiguous), a contiguous 3-channel BGR array is obtained
        frame = picam2.capture_array()[:, :, :3].copy()  
        
        # BGR frames are used for the OpenCV display
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        # Convert the BGR frame to a RGB frame for MediaPipe processing 
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        
        
        # Face detection section:
        
        
        # Resize the frame using cv_scaler to increase performance (less pixels processed, less time spent)
        resized_rgb_frame = cv2.resize(rgb_frame, (0, 0), fx=(1/cv_scaler), fy=(1/cv_scaler))
        
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(resized_rgb_frame)
        face_encodings = face_recognition.face_encodings(resized_rgb_frame, face_locations, model='small')
    
        face_names = []
        for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
        
        # Use the known face with the smallest distance to the recognised hand
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)
            
            
        # Gesture recognition section:
                
        # Process with Gesture Recognizer
        image_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        gesture_recognition_result = gesture_recognizer.recognize(image_mp)
        
        # Draw the face results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled
            top *= cv_scaler
            right *= cv_scaler
            bottom *= cv_scaler
            left *= cv_scaler
        
            # Draw a box around the face
            cv2.rectangle(bgr_frame, (left, top), (right, bottom), (244, 42, 3), 3)
        
            # Draw a label with a name below the face
            cv2.rectangle(bgr_frame, (left -3, top -35), (right +3, top), (244, 42, 3), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(bgr_frame, name, (left + 6, top - 6), font, 1.0, (255, 255, 255), 1)
        
        # Initialise an empty list to store face bounding boxes and their associated usernames   
        face_boxes = []
        for (top, right, bottom, left), name in zip(face_locations, face_names):
              # Scale back up face locations since the frame we detected in was scaled
              top *= cv_scaler
              right *= cv_scaler
              bottom *= cv_scaler
              left *= cv_scaler
              
              face_boxes.append((left, top, right, bottom, name)) # Stores the scaled coordinates and the 
                                                                  # associated username as a tuple 
        
        # Initialise an empty list in which the wrist coordinates of users' hands can be stored
        hand_position = []
        if gesture_recognition_result.hand_landmarks:
            for landmarks in gesture_recognition_result.hand_landmarks:
                wrist = landmarks[0] # Acquires the landmark of the user's wrist (this is at index 0 of Mediapipe's hand model)
                wrist_x = int(wrist.x * bgr_frame.shape[1]) # X and Y coordinates of the wrist location
                wrist_y = int(wrist.y * bgr_frame.shape[0]) # Covert normalised (between the range of 0 and 1) X and Y coordinates into pixel positions
                hand_position.append((wrist_x, wrist_y))  # Stores the wrist coordinates as a tuple
                
        # Initialise an empty list in which the wrist coordinates of users' hands can be associated with the face closest to them and stored 
        hand_face_pairs = []
        for hand_index, (hx, hy) in enumerate(hand_position):
            closest_face = None # A null variable is assigned to the closest face detected
            min_dist = float('inf')
            
            for (left, top, right, bottom, name) in face_boxes:
                face_center_x = (left + right) // 2  # The horizontal and vertical centre of face bounding boxes are calculated
                face_center_y = (top + bottom) // 2
                distance = ((hx - face_center_x)**2 + (hy - face_center_y)**2)**0.5 # The Euclidean (shortest) distance between wrists and
                # the centre of the face is calculated
                
                if distance < min_dist: # Updates the closest face if the current face detected is closer 
                    min_dist = distance # The minimum distance is the shortest distance between the wrist and centre of the face
                    closest_face = name
            hand_face_pairs.append((hand_index, closest_face)) # Stores the closest face detected and the 
                                                                  # associated index number as a tuple 
                
        # Draw the gesture results
        if gesture_recognition_result.gestures:
           for index, (gesture_group, landmarks) in enumerate(zip(gesture_recognition_result.gestures,
                                                                  gesture_recognition_result.hand_landmarks)):
              
              top_gesture = gesture_group[0]
              # Access landmarks from the list of landmarks 
              wrist = landmarks[0]
              text_x = int(wrist.x * bgr_frame.shape[1]) # X and Y coordinates of the wrist location will be used to position gesture text
              text_y = int(wrist.y * bgr_frame.shape[0]) -30 # Covert normalised (between the range of 0 and 1) X and Y
                                                             # coordinates into pixel positions 
              # The hand closest to the first detected face will be used to respond to gestures
              # with usernames 
              
              
              username = "Unknown" # The face of the user could not be identified
              if index < len(hand_face_pairs): # Executes as long as there are hand (wrist) associations in the list
                  _, username = hand_face_pairs[index] # The first element of the tuple (index) is stored in "_", it is not used
              # The second element (the username) is stored from the tuple 
    
              if top_gesture.category_name in gesture_text:
                gesture_response = gesture_text[top_gesture.category_name].format(username=username)
              else:
                gesture_response = f"{username}: {top_gesture.category_name}"
              # Display the response for a specific gesture of a specific user
              cv2.putText(bgr_frame, gesture_response, (text_x, text_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) # Green text will be used to identify each hand gesture
                                                                         # with the username of the recognised user face
          #    cv2.putText(bgr_frame, f"Hand {index + 1}: {top_gesture.category_name}", (text_x, text_y),
             #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) # Green text will be used to identify each hand gesture
             
        
        # Performance measurement section
        
        # Measuring and recording the duration of each frame to determine the fps over the entire test period
        frame_end_time = time.time()
        frame_duration = frame_end_time - frame_start_time
        if frame_duration > 0:
            current_fps = 1.0 / frame_duration
            fps_readings.append(current_fps)

        # Detection logging for analysis of a known distance
        face_detected_this_frame = len(face_names) > 0 # True if at least one face (of a registered or unknown user) was detected this frame
        gesture_detected_this_frame = bool(gesture_recognition_result.gestures) # True if list is not empty, if MediaPipe detects a least one hand in this frame

        face_detected_log.append(face_detected_this_frame)
        gesture_detected_log.append(gesture_detected_this_frame)
        both_detected_log.append(face_detected_this_frame and gesture_detected_this_frame)
        
        
        # Calculate and update Frames Per Second, display its rolling average 
        frame_count += 1
        elapsed_time = time.time() - frame_start_time
        if elapsed_time > 1:
            fps = frame_count / elapsed_time
            frame_count = 0
            frame_start_time = time.time()
            
        cv2.putText(bgr_frame, f"FPS: {fps:.1f}", (bgr_frame.shape[1] - 150, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the test mode selected 
        if RUN_TEST_MODE:
             cv2.putText(bgr_frame, "TEST MODE", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
    
    
        # Display everything over the video feed.
        cv2.imshow('Complete Recognition', bgr_frame)
        
        # Break the loop and stop the script if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            if RUN_TEST_MODE and not fps_readings: # Handles the case when "q" is pressed before first frame completes
                 print("[INFO] Test ended prematurely, no FPS data collected.")
            break

finally:
        # Display the captured performance metrics
    if RUN_TEST_MODE and fps_readings:
        print("\n--- Performance test results ---")
        avg_fps = statistics.mean(fps_readings)
        max_fps = max(fps_readings)
        min_fps = min(fps_readings)
        stdev_fps = statistics.stdev(fps_readings) if len(fps_readings) > 1 else 0

        print(f"Test duration: {TEST_DURATION_SECONDS} seconds")
        print(f"Total frames processed: {len(fps_readings)}")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Maximum FPS: {max_fps:.2f}")
        print(f"Minimum FPS: {min_fps:.2f}")
        print(f"Standard deviation FPS: {stdev_fps:.2f}")

        # Detection log statistics (useful for distance test analysis)
        total_frames = len(face_detected_log)
        face_detections = sum(face_detected_log)
        gesture_detections = sum(gesture_detected_log)
        both_detections = sum(both_detected_log)

        print("\n--- Detection log summary ---")
        print(f"Frames with face detected: {face_detections}/{total_frames} ({face_detections/total_frames*100:.1f}%)")
        print(f"Frames with gesture detected: {gesture_detections}/{total_frames} ({gesture_detections/total_frames*100:.1f}%)")
        print(f"Frames with both detected: {both_detections}/{total_frames} ({both_detections/total_frames*100:.1f}%)")

        print("\n--- Distance test guidance ---")
        print("To evaluate the effective recognition distance:")
        print("1. Run this script with RUN_TEST_MODE = True.")
        print("2. Position yourself at a known starting distance (e.g., 1 meter).")
        print("3. Test whether a face, specific gesture, or both can be detected for the duration of the test.")
        print("4. After the test finishes (or when 'q' is pressed), check the 'Detection log summary' above.")
        print("5. Correlate the performance results with the physical distance tested.")
        print("6. Analyse whether detection rates (e.g., >80-90%) were consistently high for the face, gestures, or both during the time you held that pose at that distance.")
        print("7. Change the distance (e.g., by 0.5 meters) and repeat all the previous steps to analyse the performance at a different distance.")


    elif RUN_TEST_MODE:
         print("\n--- Performance test results ---")
         print("No frames processed or test ended before completion.")
    

    
    picam2.stop()
    cv2.destroyAllWindows()
    gesture_recognizer.close()
