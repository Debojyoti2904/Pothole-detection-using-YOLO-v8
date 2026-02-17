import streamlit as st
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import time
from PIL import Image
import os
import tempfile

# --- Page Configuration ---
st.set_page_config(
    page_title="Pothole Detection App",
    page_icon="🕳️",
    layout="wide"
)

# --- Model and Tracker Initialization ---
try:
    model = YOLO('runs/detect/pothole_detector_yolov8n/weights/best.pt') 
except Exception as e:
    st.error(f"Error loading the YOLO model: {e}")
    st.info("Please make sure the model file 'best.pt' is in the correct directory.")
    st.stop()

tracker = sv.ByteTrack()
box_annotator = sv.BoxAnnotator()

# We are customizing the label annotator with a smaller text scale and padding.
label_annotator = sv.LabelAnnotator(text_scale=0.5, text_padding=5)

def callback(frame: np.ndarray, confidence: float, is_video: bool = True) -> np.ndarray:
    """
    Performs detection and tracking on a single frame and annotates it.
    The 'is_video' flag determines whether to use the tracker.
    """
    results = model(frame, conf=confidence, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    # Only update the tracker if it's a video frame.
    if is_video:
        detections = tracker.update_with_detections(detections)
        # --- THIS IS THE FIX ---
        # Create labels WITHOUT the tracker ID.
        labels = [
            f"Pothole {conf:0.2f}"
            for conf, class_id, tracker_id
            in zip(detections.confidence, detections.class_id, detections.tracker_id)
        ]
    else:
        # For single images, create simpler labels without the tracker ID.
        labels = [
            f"Pothole {conf:0.2f}"
            for conf, class_id
            in zip(detections.confidence, detections.class_id)
        ]
    # --- END OF FIX ---

    annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    return annotated_frame

def video_input(data_src, confidence):
    vid_file = None
    if data_src == 'Sample data':
        # Using the direct path to your sample video.
        vid_file = "D:\\Pothole_Detection\\sample\\video\\Screen Recording 2025-08-07 234409.mp4"
        if not os.path.exists(vid_file):
            st.error("Sample video not found at the specified path!")
            st.info(f"Please make sure the file exists at: {vid_file}")
            return
        
    else: # 'Upload your own data'
        vid_bytes = st.sidebar.file_uploader("Upload a video", type=['mp4', 'mov', 'avi', 'mkv'])
        if vid_bytes:
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(vid_bytes.read())
            vid_file = tfile.name
        else:
            return

    if vid_file:
        cap = cv2.VideoCapture(vid_file)
        
        # Close the temporary file handle immediately after creating the VideoCapture object
        if data_src == 'Upload your own data' and 'tfile' in locals():
            tfile.close()

        # Only set up saving for uploaded files, not the sample
        if data_src == 'Upload your own data':
            results_folder = "D:\\Pothole_Detection\\results\\videos"
            if not os.path.exists(results_folder):
                os.makedirs(results_folder)

            input_filename = vid_bytes.name
            base_name, ext = os.path.splitext(input_filename)
            output_path = os.path.join(results_folder, f"{base_name}_result{ext}")

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_in = cap.get(cv2.CAP_PROP_FPS)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps_in, (width, height))
        
        st.markdown("---")
        st.info("Processing video... Please wait.")
        output_placeholder = st.empty()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Tell the callback it's a video frame
            processed_frame = callback(frame, confidence, is_video=True)
            output_placeholder.image(processed_frame, channels="BGR", use_container_width=True)
            
            # Only write to file if it's an uploaded video
            if data_src == 'Upload your own data':
                out.write(processed_frame)

        cap.release()
        
        # Finalize and clean up only for uploaded videos
        if data_src == 'Upload your own data':
            out.release()
            os.remove(vid_file)
            st.success(f"Video processing complete! Result saved to '{output_path}'")
            with open(output_path, "rb") as file:
                st.download_button(
                    label="Download Processed Video",
                    data=file,
                    file_name=os.path.basename(output_path),
                    mime="video/mp4"
                )
        else:
            st.success("Sample video processing complete!")

def image_input(data_src, confidence):
    img_file = None
    if data_src == 'Sample data':
        img_file = os.path.join("sample", "image", "pothole.jpeg")
        if not os.path.exists(img_file):
            st.error("Sample image 'pothole.jpeg' not found!")
            return
    else:
        img_bytes = st.sidebar.file_uploader("Upload an image", type=['png', 'jpeg', 'jpg'])
        if img_bytes:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.png') 
            tfile.write(img_bytes.read())
            img_file = tfile.name
        else:
            return

    if img_file:
        image = cv2.imread(img_file)
        if image is None:
            st.error(f"Could not read the image file: {img_file}")
            return
            
        # Close the temporary file if it was created
        if data_src == 'Upload your own data' and 'tfile' in locals():
            tfile.close()

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", channels="BGR", use_container_width=True)
        with col2:
            # Tell the callback it's a single image
            processed_image = callback(image, confidence, is_video=False)
            st.image(processed_image, caption="Pothole Detection Result", channels="BGR", use_container_width=True)

            results_folder = "D:\\Pothole_Detection\\results\\images"
            if not os.path.exists(results_folder):
                os.makedirs(results_folder)

            input_filename = os.path.basename(img_file) if data_src == 'Sample data' else img_bytes.name
            base_name, ext = os.path.splitext(input_filename)
            output_path = os.path.join(results_folder, f"{base_name}_result{ext}")
            
            cv2.imwrite(output_path, processed_image)
            st.success(f"Processed image saved to '{output_path}'")

            with open(output_path, "rb") as file:
                st.download_button(
                    label="Download Processed Image",
                    data=file,
                    file_name=os.path.basename(output_path),
                    mime="image/png"
                )
        
        if data_src == 'Upload your own data':
            os.remove(img_file)

def main():
    st.title("Pothole Detection")
    st.sidebar.title("Settings")
    input_type = st.sidebar.radio("Select input type:", ['Image', 'Video'])
    input_source = st.sidebar.radio("Select input source:", ['Sample data', 'Upload your own data'])
    confidence = st.sidebar.slider('Confidence Threshold', min_value=0.1, max_value=1.0, value=0.3)

    if input_type == 'Video':
        video_input(input_source, confidence)
    else:
        image_input(input_source, confidence)

if __name__ == "__main__":
    main()
