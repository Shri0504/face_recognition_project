import os
import cv2
import face_recognition
from django.shortcuts import render

# Path to the media directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEDIA_DIR = os.path.join(BASE_DIR, 'media/images')

# Load and encode images
known_names = ["Ratan Tata", "Elon Musk", "Mark Zuckerberg"]
image_paths = [
    os.path.join(MEDIA_DIR, "ratan_tata.png"),
    os.path.join(MEDIA_DIR, "elon_musk.png"),
    os.path.join(MEDIA_DIR, "mark_zuckerberg.png")
]

known_encodings = [face_recognition.face_encodings(face_recognition.load_image_file(img))[0] for img in image_paths]

# View for rendering the home page
def index(request):
    return render(request, 'recognition/index.html')

# View for starting face recognition
def recognize_faces(request):
    camera = cv2.VideoCapture(0)  # Open webcam
    face_detection_model = "hog"
    tolerance = 0.6  # Recognition tolerance

    while True:
        ret, frame = camera.read()

        face_locations = face_recognition.face_locations(frame, model=face_detection_model)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"

            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)

            if True in matches:
                best_match_index = matches.index(True)
                if face_distances[best_match_index] <= tolerance:
                    name = known_names[best_match_index]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1)

        cv2.imshow('Face Detection and Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()
    return render(request, 'recognition/index.html')
