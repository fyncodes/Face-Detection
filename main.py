import cv2
import face_recognition
import os
import pickle
import numpy as np

# File to store known face encodings
KNOWN_FACES_FILE = "known_faces.pkl"

# Load known faces if the file exists, or initialize an empty dictionary
if os.path.exists(KNOWN_FACES_FILE):
    with open(KNOWN_FACES_FILE, "rb") as file:
        known_faces = pickle.load(file)
        # Ensure encodings are loaded as NumPy arrays
        known_faces = {name: np.array(encoding) for name, encoding in known_faces.items()}
else:
    known_faces = {}

print("Loaded known faces:", known_faces.keys())  # Debug: Show loaded face names

# Initialize webcam
cap = cv2.VideoCapture(0)

# Set to track recognized faces in the current session
recognized_faces = set()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces and get encodings
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Check if the face is already recognized in this session
        matches = face_recognition.compare_faces(list(known_faces.values()), face_encoding, tolerance=0.6)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = list(known_faces.keys())[match_index]

            # If the face is already recognized, skip processing
            if name in recognized_faces:
                continue

            # Mark the face as recognized
            recognized_faces.add(name)
            print(f"Matched: {name}")  # Debug: Show matched face name
        else:
            # New face detected, ask for a name
            name = input("New face detected. Enter the person's name: ").strip()
            if name:  # Save only if a valid name is provided
                known_faces[name] = face_encoding
                print(f"Added new face: {name}")  # Debug: Confirm new face added

        # Draw a box around the face and label it
        top, right, bottom, left = [v * 4 for v in face_location]  # Scale back up
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Face Recognition", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Save known faces to the file when exiting
with open(KNOWN_FACES_FILE, "wb") as file:
    pickle.dump(known_faces, file)

print("Current saved faces:", known_faces.keys())  # Debug: Show saved face names before exit

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()