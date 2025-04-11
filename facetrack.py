import cv2
import mediapipe as mp
import numpy as np
import uuid

# Initialize MediaPipe Face Mesh module
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Start video capture (webcam or a video file)
cap = cv2.VideoCapture("inputvids/rhettandlink_edit3.mp4")

# Initialize face mesh model
with mp_face_mesh.FaceMesh(refine_landmarks=True,
                           max_num_faces=2,
                           min_detection_confidence=0.1,
                           min_tracking_confidence=0.5) as face_mesh:

    prev_face_positions = []  # To store positions of faces from the previous frame
    face_ids = {}  # Dictionary to store face IDs and their bounding boxes

    def generate_id():
        """Generate a unique ID for each face."""
        return str(uuid.uuid4())[:8]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or frame read error.")
            break

        # Convert the frame to RGB (required for MediaPipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to find face mesh landmarks
        results = face_mesh.process(rgb_frame)

        current_face_positions = []
        current_face_ids = []

        if results.multi_face_landmarks:
            # Extract bounding boxes for faces
            for i, face_landmarks in enumerate(results.multi_face_landmarks):
                # Get the coordinates of the landmarks
                h, w, _ = frame.shape
                x_min, y_min, x_max, y_max = w, h, 0, 0

                # Iterate through the face landmarks to find the min/max bounding box coordinates
                for landmark in face_landmarks.landmark:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)

                # Store the bounding box for the current face
                current_face_positions.append((x_min, y_min, x_max, y_max))

                # Compare with previous positions to find matching faces
                matched_id = None
                for prev_id, (prev_x_min, prev_y_min, prev_x_max, prev_y_max) in face_ids.items():
                    # Calculate the distance between the current face and the previous one
                    distance = np.sqrt((prev_x_min - x_min) ** 2 + (prev_y_min - y_min) ** 2)

                    # If the distance is small, consider it the same face
                    if distance < 100:  # Threshold for matching faces (can be adjusted)
                        matched_id = prev_id
                        break

                if matched_id is None:
                    # If no match is found, assign a new ID
                    matched_id = generate_id()

                # Add the matched ID to the list of current face IDs
                current_face_ids.append(matched_id)

                # Update the dictionary with the new or matched face ID and position
                face_ids[matched_id] = (x_min, y_min, x_max, y_max)

                # Draw the bounding box and overlay the face ID
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {matched_id}", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Cleanup: Remove IDs of faces not found in the current frame
        prev_face_positions_set = set(prev_face_positions)
        current_face_positions_set = set(current_face_positions)

        ids_to_remove = prev_face_positions_set - current_face_positions_set
        for prev_position in ids_to_remove:
            # Find and remove the corresponding ID by matching the bounding box position
            face_ids = {key: value for key, value in face_ids.items() if value != prev_position}

        prev_face_positions = current_face_positions  # Update previous face positions for the next frame

        # Show the frame with face mesh landmarks and bounding boxes with IDs
        cv2.imshow('Face Mesh Tracking', frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

