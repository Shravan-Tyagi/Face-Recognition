# FACE RECOGNITION
# ADD IMAGES OF FACES TO BE RECOGNIZED IN known_faces


import face_recognition
import numpy as np
from PIL import Image, ImageOps
import pickle
import cv2
import os


known_face_encodings = []
known_face_names = []


# Load image data
def load_known_faces():
    invalid_images = []
    print(1)

    for filename in os.listdir("known_faces"):
        print(2)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.jfif', '.webp')):
            image_path = f"known_faces/{filename}"
            print(3)

            print(filename)

            # Load and auto rotate
            image_pil = Image.open(image_path)
            image_pil = ImageOps.exif_transpose(image_pil)
            image = np.array(image_pil)

            # Improve contrast
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            image = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)


            # Try HOG then CNN
            face_locations = face_recognition.face_locations(image, model="hog")
            if not face_locations:
                face_locations = face_recognition.face_locations(image, model="cnn")


            # Skip very small faces
            #face_locations = [loc for loc in face_locations if (loc[2] - loc[0]) >= 40]

            if face_locations:
                encodings = face_recognition.face_encodings(image, face_locations)
                for encoding in encodings:
                    known_face_encodings.append(encoding)
                    known_face_names.append(os.path.splitext(filename)[0])
            else:
                invalid_images.append(filename)

        # Save to cache
    with open("face_cache.pkl", "wb") as f:
        pickle.dump((known_face_encodings, known_face_names), f)

    return invalid_images



def recognize(image, model=None):
    if not os.path.exists("face_cache.pkl"):
        return None, None

    with open("face_cache.pkl", "rb") as f:
        known_face_encodings, known_face_names = pickle.load(f)

    try:
        frame = cv2.imread(image)

        # Resize image
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        if model == "hog":
            face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
        else:
            face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
            if not face_locations:
                face_locations = face_recognition.face_locations(rgb_small_frame, model="cnn")

        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Compare with known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = face_distances.argmin()
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    name = "".join(i for i in name if i.isalpha())

            face_names.append(name)

        # Display results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            # Get text size and baseline
            font_scale = 0.8
            font_thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                                                  font_thickness)

            # Calculate box position dynamically
            box_top = bottom
            box_bottom = bottom + text_height + baseline + 2
            box_left = left
            box_right = left + text_width + 10  # extra padding

            if text_width < right - left:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (box_left, box_top), (right, box_bottom), (0, 255, 0), cv2.FILLED)
            else:
                cv2.rectangle(frame, (left, top), (box_right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (box_left, box_top), (box_right, box_bottom), (0, 255, 0), cv2.FILLED)


            # Put text inside
            cv2.putText(frame, name, (left + 5, bottom + text_height + 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (255, 255, 255), font_thickness)


        cv2.imwrite("recognized_image.jpg", frame)
        return "recognized_image.jpg", face_names

    except Exception as e:
        return None, None


if __name__ == "__main__":
    #load_known_faces()
    image_, faces = recognize("known_faces/diddy.jfif")
    print(faces)
