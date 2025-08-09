import cv2
import os

def preprocess_face_image(image_path, output_size=(48, 48)):
    """
    Detect, crop and resize face in the image.
    Args:
        image_path (str): Path to the image file.
        output_size (tuple): Desired output size for the face image.
    Returns:
        image (np.ndarray): Preprocessed face image array, or None if no face found.
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    img = cv2.imread(image_path)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print(f"No face detected in {image_path}")
        return None

    # Assuming first detected face
    (x,y,w,h) = faces[0]
    face_img = gray_img[y:y+h, x:x+w]

    # Resize to output_size
    face_img = cv2.resize(face_img, output_size)

    # Normalize pixel values to range 0-1
    face_img = face_img / 255.0

    return face_img

if __name__ == "__main__":
    img_path = 'path_to_face_image.jpg'
    face_img = preprocess_face_image(img_path)
    if face_img is not None:
        print(f"Preprocessed face image shape: {face_img.shape}")
