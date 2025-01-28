from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO('yolo11n.pt')

# Video file path
video_path = "highway.mp4"  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Create a named window that allows resizing
cv2.namedWindow('Car Detection in Video', cv2.WINDOW_NORMAL)

# Entry and exit counters
entry_count = 0
exit_count = 0

# Horizontal line properties for tracking
line_y = 500  # Y-coordinate of the line (adjustable)
line_start_x = 200  # Starting x-coordinate of the line
line_end_x = 1000  # Ending x-coordinate of the line

# Dictionary to track the previous positions of tracked objects
tracked_objects = {}

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break

    # Detect and track objects
    results = model.track(frame, persist=True)

    # Extract detections for cars only (class 2)
    car_boxes = []
    for result in results:
        for box in result.boxes:
            obj_id = box.id
            if obj_id is not None and int(box.cls) == 2:  # Check if ID exists and class is 'car'
                car_boxes.append((int(obj_id), box.xyxy[0]))  # Store ID and bounding box

    # Process each detected car box
    for obj_id, box in car_boxes:
        x1, y1, x2, y2 = map(int, box)
        cx = (x1 + x2) // 2  # Center x of the bounding box
        cy = (y1 + y2) // 2  # Center y of the bounding box

        # Draw the bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Track the object by its ID
        if obj_id not in tracked_objects:
            tracked_objects[obj_id] = (cx, cy)  # Initialize with the current position
        else:
            prev_cx, prev_cy = tracked_objects[obj_id]

            # Determine crossing direction
            if prev_cy < line_y and cy >= line_y:  # Crossed from top to bottom (entry)
                exit_count+=1
                print(f"Car {obj_id} entered. Entry count: {exit_count}")
            elif prev_cy >= line_y and cy < line_y:  # Crossed from bottom to top (exit)
                entry_count+=1
                print(f"Car {obj_id} exited. Exit count: {entry_count}")

            # Update the tracked position
            tracked_objects[obj_id] = (cx, cy)

    # Draw the horizontal line for entry/exit tracking
    cv2.line(frame, (line_start_x, line_y), (line_end_x, line_y), (0, 0, 255), 2)

    # Display entry and exit counts on the frame in red color
    cv2.putText(frame, f"Entries: {entry_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Exits: {exit_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow('Car Detection in Video', frame)

    # Handle keypress events
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit on 'q'
        break

# Release the video file and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
