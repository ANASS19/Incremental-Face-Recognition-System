from flask import Flask, render_template, Response, request, redirect, url_for, flash
import cv2
import torch
import numpy as np
import os
from facenet_pytorch import InceptionResnetV1, MTCNN
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

# Initializing Flask App
app = Flask(__name__)
app.secret_key = 'supersecretkey'  # For flash messages

# uisng GPU if availabile
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Loading Pretrained FaceNet Model
facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Loading Face Detector (MTCNN)
mtcnn = MTCNN(keep_all=True, device="cpu")

# Load or Initialize Embeddings & Labels
if os.path.exists("face_embeddings.npz"):
    data = np.load("face_embeddings.npz", allow_pickle=True)
    embeddings = list(data["embeddings"])
    labels = list(data["labels"])
else:
    embeddings, labels = [], []

# Encoding Labels
label_encoder = LabelEncoder()
label_encoder.fit(labels)

# Defining Classifier Model
class FaceClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FaceClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# Loaing Classifier and Fix Size Mismatch
def load_classifier():
    global classifier

    num_classes = len(np.unique(labels))
    classifier = FaceClassifier(input_size=512, num_classes=num_classes).to(device)

    if os.path.exists("face_classifier.pth"):
        classifier.load_state_dict(torch.load("face_classifier.pth"))
    classifier.eval()

load_classifier()

# Face Preprocessing
def preprocess_face(img):
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    elif isinstance(img, str):
        img = Image.open(img).convert("RGB")

    img_tensor = transform(img).unsqueeze(0).to(device)
    return img_tensor

# Extract Face Embedding
def extract_embedding(img):
    img_tensor = preprocess_face(img)
    with torch.no_grad():
        embedding = facenet_model(img_tensor).cpu().numpy().flatten()
    return embedding

# Update Classifier
def update_classifier():
    global classifier

    num_classes = len(np.unique(labels))
    updated_classifier = FaceClassifier(input_size=512, num_classes=num_classes).to(device)

    new_state_dict = updated_classifier.state_dict()
    old_state_dict = classifier.state_dict() if hasattr(classifier, 'state_dict') else {}

    for key in old_state_dict.keys():
        if "fc3" not in key:
            new_state_dict[key] = old_state_dict[key]

    updated_classifier.load_state_dict(new_state_dict)
    classifier = updated_classifier
    torch.save(classifier.state_dict(), "face_classifier.pth")

# Enroll a New Student via Webcam
def enroll_new_student(student_name, frame):
    global embeddings, labels

    # Detect face in the captured frame
    boxes, _ = mtcnn.detect(frame)
    if boxes is not None:
        for box in boxes:
            x, y, w, h = map(int, box)
            face = frame[y:h, x:w]

            # Extract embedding and update dataset
            new_embedding = extract_embedding(face)
            embeddings.append(new_embedding)
            labels.append(student_name)

            # Saveing updated embeddings & labels
            np.savez_compressed("face_embeddings.npz", 
                    embeddings=np.array(embeddings, dtype=np.float32), 
                    labels=np.array(labels, dtype=str))  # ✅ Change dtype from object → str


            update_classifier()
            flash(f"✅ Student '{student_name}' enrolled successfully!", "success")
            return True  # Enrollment successful
    else:
        flash("⚠️ No face detected. Please try again!", "danger")
        return False  # Enrollment failed

# Recognizing Face in Frame
def recognize_face(frame):
    boxes, _ = mtcnn.detect(frame)

    if boxes is not None:
        for box in boxes:
            x, y, w, h = map(int, box)
            face = frame[y:h, x:w]

            new_embedding = extract_embedding(face)
            similarities = [np.linalg.norm(new_embedding - emb) for emb in embeddings]
            min_distance = min(similarities)
            min_index = similarities.index(min_distance)
            threshold = 0.8

            if min_distance < threshold:
                return labels[min_index]
    return "Unknown"

# Real-Time Face Recognition
def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (640, 480))
        predicted_name = recognize_face(frame)

        text = f"Acces Granted: {predicted_name}" if predicted_name != "Unknown" else " Access Denied"
        color = (0, 255, 0) if predicted_name != "Unknown" else (0, 0, 255)

        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# Routes for Flask App
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Enroll Student via Webcam Capture
@app.route('/capture_enroll', methods=['POST'])
def capture_enroll():
    student_name = request.form["name"].strip()

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if ret:
        success = enroll_new_student(student_name, frame)
        if success:
            flash(f"✅ Student '{student_name}' enrolled successfully!", "success")
        else:
            flash("⚠️ Enrollment failed. No face detected!", "danger")
    else:
        flash("⚠️ Capture failed!", "danger")

    return redirect(url_for("index"))

# Start Flask App
if __name__ == "__main__":
    app.run(debug=True)






# create by anass nassiri 908475 for ict EXam.
#Thanks