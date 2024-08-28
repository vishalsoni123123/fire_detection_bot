
# #################      code with arduino integration   ##############################
import serial
import time
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Initialize serial communication with Arduino
arduino_serial = serial.Serial('COM17', 9600)  # Change COM port as needed

# Load the trained model
model = YOLO('best.pt')

# Check if CUDA is available and use GPU if possible
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Inference on a live camera feed
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

def create_mask_for_fire(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([0, 50, 50])
    upper_hsv = np.array([10, 255, 255])
    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def map_coordinates_to_servo(mapped_x, mapped_y):
    # Invert the X-coordinate to correct the mirrored mapping
    inverted_x = -mapped_x
    
    # Map coordinates from (-255, 255) to servo angles (0, 180)
    servoX = int(np.interp(inverted_x, [-255, 255], [50,130]))
    servoY = int(np.interp(mapped_y, [-255, 255], [60,120]))
    
    return servoX, servoY

def send_to_arduino(servoX, servoY):
    # Send the servo angles to Arduino
    command = f"{servoX},{servoY}\n"
    arduino_serial.write(command.encode())

def apply_fire_detection_and_spotting(frame, bbox):
    x1, y1, x2, y2 = bbox
    roi = frame[y1:y2, x1:x2]

    mask = create_mask_for_fire(roi)
    fire_roi = cv2.bitwise_and(roi, roi, mask=mask)
    gray_mask = cv2.cvtColor(fire_roi, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 50:
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            cx, cy = int(cx) + x1, int(cy) + y1

            # Convert the coordinates to the mapped frame
            mapped_x = int((cx - frame.shape[1] // 2) * (510 / frame.shape[1]))
            mapped_y = int((cy - frame.shape[0] // 2) * (510 / frame.shape[0]))
            
            # Send the mapped coordinates to Arduino
            servoX, servoY = map_coordinates_to_servo(mapped_x, mapped_y)
            send_to_arduino(servoX, servoY)

            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"({servoX},{servoY})", (cx, cy), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            print(f"({servoX},{servoY})")

    frame[y1:y2, x1:x2] = cv2.addWeighted(frame[y1:y2, x1:x2], 0.5, fire_roi, 0.5, 0)
    
    return frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = results[0].plot()

    fire_detected = False
    for result in results[0].boxes.data:
        confidence = result[4]
        class_id = int(result[5])
        if confidence > 0.75 and class_id == 0:
            fire_detected = True
            bbox = result[:4].cpu().numpy().astype(int)
            break

    if fire_detected:
        start_time = time.time()
        while time.time() - start_time < 20:
            frame_with_spots = apply_fire_detection_and_spotting(frame, bbox)
            cv2.imshow('Fire Detection with Servo Control', frame_with_spots)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        cv2.imshow('YOLOv8 Live Inference', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


#######################################4
import serial
import time
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Initialize serial communication with Arduino
arduino_serial = serial.Serial('COM17', 9600)  # Change COM port as needed

# Load the trained model
model = YOLO('best.pt')

# Check if CUDA is available and use GPU if possible
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Inference on a live camera feed
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

def create_mask_for_fire(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([0, 50, 50])
    upper_hsv = np.array([10, 255, 255])
    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def map_coordinates_to_servo(mapped_x, mapped_y):
    # Invert the X-coordinate to correct the mirrored mapping
    inverted_x = -mapped_x
    
    # Map coordinates from (-255, 255) to servo angles (0, 180)
    servoX = int(np.interp(inverted_x, [-255, 255], [0, 180]))
    servoY = int(np.interp(mapped_y, [-255, 255], [0, 180]))
    
    return servoX, servoY

def send_to_arduino(servoX, servoY):
    # Send the servo angles to Arduino
    command = f"{servoX},{servoY}\n"
    arduino_serial.write(command.encode())

def apply_fire_detection_and_spotting(frame, bbox):
    x1, y1, x2, y2 = bbox
    roi = frame[y1:y2, x1:x2]

    mask = create_mask_for_fire(roi)
    fire_roi = cv2.bitwise_and(roi, roi, mask=mask)
    gray_mask = cv2.cvtColor(fire_roi, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 50:
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            cx, cy = int(cx) + x1, int(cy) + y1

            # Convert the coordinates to the mapped frame
            mapped_x = int((cx - frame.shape[1] // 2) * (510 / frame.shape[1]))
            mapped_y = int((cy - frame.shape[0] // 2) * (510 / frame.shape[0]))
            
            # Send the mapped coordinates to Arduino
            servoX, servoY = map_coordinates_to_servo(mapped_x, mapped_y)
            send_to_arduino(servoX, servoY)

            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"({servoX},{servoY})", (cx, cy), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            print(f"({servoX},{servoY})")

    frame[y1:y2, x1:x2] = cv2.addWeighted(frame[y1:y2, x1:x2], 0.5, fire_roi, 0.5, 0)
    
    return frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = results[0].plot()

    fire_detected = False
    for result in results[0].boxes.data:
        confidence = result[4]
        class_id = int(result[5])
        if confidence > 0.75 and class_id == 0:
            fire_detected = True
            bbox = result[:4].cpu().numpy().astype(int)
            break

    if fire_detected:
        start_time = time.time()
        while time.time() - start_time < 20:
            frame_with_spots = apply_fire_detection_and_spotting(frame, bbox)
            cv2.imshow('Fire Detection with Servo Control', frame_with_spots)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        cv2.imshow('YOLOv8 Live Inference', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


#############################################################################################################################################



# import time
# import cv2
# import torch
# import numpy as np
# from ultralytics import YOLO

# # Load the trained model
# model = YOLO('best.pt')

# # Check if CUDA is available and use GPU if possible
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model.to(device)

# # Inference on a live camera feed
# cap = cv2.VideoCapture(0)  # Use 0 for the default camera

# def create_mask_for_fire(image):
#     image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     lower_hsv = np.array([0, 50, 50])
#     upper_hsv = np.array([10, 255, 255])
#     mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#     return mask

# def apply_fire_detection_and_spotting(frame, bbox):
#     x1, y1, x2, y2 = bbox
#     roi = frame[y1:y2, x1:x2]

#     mask = create_mask_for_fire(roi)
#     fire_roi = cv2.bitwise_and(roi, roi, mask=mask)
#     gray_mask = cv2.cvtColor(fire_roi, cv2.COLOR_BGR2GRAY)
#     contours, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     for contour in contours:
#         if cv2.contourArea(contour) > 50:
#             (cx, cy), radius = cv2.minEnclosingCircle(contour)
#             cx, cy = int(cx) + x1, int(cy) + y1

#             # Convert the coordinates to the mapped frame
#             mapped_x = int((cx - frame.shape[1] // 2) * (510 / frame.shape[1]))
#             mapped_y = int((cy - frame.shape[0] // 2) * (510 / frame.shape[0]))
            
#             # Display the mapped coordinates on the frame
#             cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
#             cv2.putText(frame, f"({mapped_x},{mapped_y})", (cx, cy), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
#             print(f"({mapped_x},{mapped_y})")

#     frame[y1:y2, x1:x2] = cv2.addWeighted(frame[y1:y2, x1:x2], 0.5, fire_roi, 0.5, 0)
    
#     return frame

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     results = model(frame)
#     annotated_frame = results[0].plot()

#     fire_detected = False
#     for result in results[0].boxes.data:
#         confidence = result[4]
#         class_id = int(result[5])
#         if confidence > 0.75 and class_id == 0:
#             fire_detected = True
#             bbox = result[:4].cpu().numpy().astype(int)
#             break

#     if fire_detected:
#         start_time = time.time()
#         while time.time() - start_time < 5:
#             frame_with_spots = apply_fire_detection_and_spotting(frame, bbox)
#             cv2.imshow('Fire Detection with Coordinates', frame_with_spots)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#     else:
#         cv2.imshow('YOLOv8 Live Inference', annotated_frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


##############################################################################################################################











# import cv2
# import torch
# import numpy as np
# from ultralytics import YOLO

# # Load the MiDaS model for depth estimation
# model_type = "DPT_Large"  # Options: "DPT_Large", "DPT_Hybrid", "MiDaS_small"
# midas = torch.hub.load("intel-isl/MiDaS", model_type)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# midas.to(device).eval()

# # Load the transform to convert input to the model format
# transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

# # Load the YOLO model for fire detection
# model = YOLO('best.pt')
# model.to(device)

# # Function to calculate the distance of fire from the camera using depth map
# def calculate_distance(depth_map, bbox):
#     # Extract the region of interest (fire) from the depth map
#     x1, y1, x2, y2 = bbox
#     roi_depth = depth_map[y1:y2, x1:x2]

#     # Calculate the average depth within the ROI
#     avg_depth = np.mean(roi_depth)
    
#     return avg_depth

# # Function to apply fire detection and calculate distance
# def apply_fire_detection_and_distance(frame, bbox):
#     # Transform the frame for depth estimation
#     input_batch = transform(frame).to(device)

#     with torch.no_grad():
#         prediction = midas(input_batch)
#         prediction = torch.nn.functional.interpolate(
#             prediction.unsqueeze(1),
#             size=frame.shape[:2],
#             mode="bicubic",
#             align_corners=False,
#         ).squeeze()
#         depth_map = prediction.cpu().numpy()

#     # Calculate distance of the fire
#     distance = calculate_distance(depth_map, bbox)
#     print(f"Distance to fire: {distance:.2f} meters")

#     # Annotate the frame with the distance
#     x1, y1, x2, y2 = bbox
#     cv2.putText(frame, f"Distance: {distance:.2f} meters", (x1, y1 - 10), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

#     return frame

# # Inference on a live camera feed
# cap = cv2.VideoCapture(0)  # Use 0 for the default camera

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Apply YOLO model to detect fire
#     results = model(frame)
#     bbox = None

#     for result in results[0].boxes.data:
#         confidence = result[4]
#         class_id = int(result[5])
#         if confidence > 0.75 and class_id == 0:
#             bbox = result[:4].cpu().numpy().astype(int)
#             break

#     if bbox is not None:
#         # Apply fire detection and distance estimation
#         frame_with_distance = apply_fire_detection_and_distance(frame, bbox)
#         cv2.imshow('Fire Detection with Distance Estimation', frame_with_distance)
#     else:
#         annotated_frame = results[0].plot()
#         cv2.imshow('YOLOv8 Live Inference', annotated_frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
