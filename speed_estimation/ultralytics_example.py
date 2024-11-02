import streamlit as st
import tempfile
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from collections import defaultdict, deque

# Default Constants
DEFAULT_SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])
DEFAULT_TARGET_HEIGHT = 250
TARGET_WIDTH = 25

VEHICLE_CLASSES = {
    0: "van",
    1: "car",
    2: "truck",
    3: "bus"
}

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

def process_video(input_video_path, output_video_path, progress_bar, status_text, source_coords, target_height):
    # Update TARGET based on input target_height
    TARGET = np.array([[0, 0], [TARGET_WIDTH - 1, 0], 
                      [TARGET_WIDTH - 1, target_height - 1], [0, target_height - 1]])
    
    # Load Model
    model = YOLO("best (2).pt")
    
    # Get video properties
    cap = cv2.VideoCapture(input_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # Create video writer with web-compatible codec
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Load video information and set up annotations
    video_info = sv.VideoInfo.from_video_path(video_path=input_video_path)
    byte_track = sv.ByteTrack(frame_rate=video_info.fps, track_activation_threshold=0.3)
    
    # Annotations and settings
    thickness = sv.calculate_optimal_line_thickness(video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(video_info.resolution_wh)
    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness, text_position=sv.Position.BOTTOM_CENTER)
    trace_annotator = sv.TraceAnnotator(thickness=thickness, trace_length=video_info.fps * 2, position=sv.Position.BOTTOM_CENTER)
    
    # Frame generator and view transformer
    frame_generator = sv.get_video_frames_generator(source_path=input_video_path)
    polygon_zone = sv.PolygonZone(polygon=source_coords)
    view_transformer = ViewTransformer(source=source_coords, target=TARGET)
    
    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))
    
    frame_idx = 0
    for frame in frame_generator:
        # Update progress
        progress = (frame_idx + 1) / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_idx + 1} of {total_frames} ({int(progress * 100)}%)")
        
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[detections.confidence > 0.3]
        detections = detections[polygon_zone.trigger(detections)]
        detections = detections.with_nms(threshold=0.5)
        detections = byte_track.update_with_detections(detections=detections)
        
        # Transform points and calculate speed
        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        points = view_transformer.transform_points(points=points).astype(int)
        
        for tracker_id, [_, y] in zip(detections.tracker_id, points):
            coordinates[tracker_id].append(y)

        labels = []
        for tracker_id, class_id in zip(detections.tracker_id, detections.class_id):
            # Get vehicle type from our dictionary
            vehicle_type = VEHICLE_CLASSES[class_id]
            
            if len(coordinates[tracker_id]) < video_info.fps / 2:
                labels.append(f"{vehicle_type}")
            else:
                coordinate_start = coordinates[tracker_id][-1]
                coordinate_end = coordinates[tracker_id][0]
                distance = abs(coordinate_start - coordinate_end)
                time = len(coordinates[tracker_id]) / video_info.fps
                speed = distance / time * 3.6
                labels.append(f"{vehicle_type} {int(speed)} km/h")

        # Annotate frame
        annotated_frame = frame.copy()
        annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        
        # Write frame
        out.write(annotated_frame)
        frame_idx += 1
    
    out.release()

# Streamlit Interface
def main():
    st.title("Vehicle Speed Detection")

    # Coordinate Input Section
    st.subheader("Detection Zone Configuration")
    use_custom_coords = st.checkbox("Use Custom Coordinates", False)

    if use_custom_coords:
        st.write("Enter the four corners of your detection zone (x, y coordinates):")
        col1, col2 = st.columns(2)
        
        with col1:
            x1 = st.number_input("Point X1", value=1252)
            y1 = st.number_input("Point Y1", value=787)
            
            x2 = st.number_input("Point X2", value=2298)
            y2 = st.number_input("Point Y2", value=803)
        
        with col2:
            x3 = st.number_input("Point X3", value=5039)
            y3 = st.number_input("Point Y3", value=2159)
            
            x4 = st.number_input("Point X4", value=-550)
            y4 = st.number_input("Point Y4", value=2159)
        
        source_coords = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    else:
        source_coords = DEFAULT_SOURCE

    # Target Height Input
    target_height = st.number_input("Distance(m)", 
                                  value=DEFAULT_TARGET_HEIGHT,
                                  min_value=50,
                                  max_value=1000,
                                  help="Height of the transformed detection zone in pixels")

    # Video File Upload
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

    if uploaded_file:
        # Create temporary files for input and output
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_input:
            tmp_input.write(uploaded_file.read())
            input_path = tmp_input.name
            
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        
        try:
            # Create progress bar and status text
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process the video with user inputs or defaults
            process_video(
                input_video_path=input_path,
                output_video_path=output_path,
                progress_bar=progress_bar,
                status_text=status_text,
                source_coords=source_coords,
                target_height=target_height
            )
            
            # Clear progress bar and status
            progress_bar.empty()
            status_text.empty()
            
            # Display success message and video
            st.success("Processing complete!")
            st.video(output_path)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()