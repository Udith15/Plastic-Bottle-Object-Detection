import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
import shutil
import platform

# Load the pre-trained YOLO model
model = YOLO(os.path.join("model", "trainedweights.pt"))
is_local = platform.system() != 'Linux'  # True if running locally

# Define output directory
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def detect_and_display_image(image_path):
    image = cv2.imread(image_path)
    results = model.predict(image)

    for result in results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        conf = result.conf[0]
        cls = int(result.cls[0])
        label = f"{model.names[cls]} {conf:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    output_image_path = os.path.join(output_dir, os.path.basename(image_path))
    output_image_path = output_image_path[:-4] + "_detected.jpg"  # Change extension if needed
    cv2.imwrite(output_image_path, image)
    return output_image_path, image

def detect_and_display_video(video_path):
    temp_video_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_video_output.close()
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_output.name, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame)

        for result in results[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            conf = result.conf[0]
            cls = int(result.cls[0])
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()

    output_video_path = os.path.join(output_dir, os.path.basename(video_path))
    output_video_path_with_extension = output_video_path + ".mp4"
    shutil.move(temp_video_output.name, output_video_path_with_extension)
    return output_video_path_with_extension

def detect_and_display_realtime():
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame)

        for result in results[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            conf = result.conf[0]
            cls = int(result.cls[0])
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB")

    cap.release()
    cv2.destroyAllWindows()

st.title("Plastic Bottle Detection")

option = st.selectbox(
    "Choose an option",
    ("Image Detection", "Video Detection", "Real-time Detection")
)

if option == "Image Detection":
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp.write(uploaded_file.getvalue())
            image_path = temp.name

        result_image_path, result_image = detect_and_display_image(image_path)
        st.image(result_image, channels="BGR")
        st.write("Output saved as:", result_image_path)

elif option == "Video Detection":
    uploaded_file = st.file_uploader("Upload a video...", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp.write(uploaded_file.getvalue())
            video_path = temp.name

        result_video_path = detect_and_display_video(video_path)
        st.write("Output saved as:", result_video_path)

        st.video(result_video_path)


elif option == "Real-time Detection":
    if is_local:
        st.write("Starting real-time detection...")
        detect_and_display_realtime()
    else:
        st.warning("Real-time detection not supported on web deployment.")

