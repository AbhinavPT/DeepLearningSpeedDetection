import argparse
from collections import defaultdict, deque

import os
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

import supervision as sv

TARGET_WIDTH = 6
TARGET_HEIGHT = 50

TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)

VEHICLE_CLASSES = {
    0: "van",
    1: "car",
    2: "truck",
    3: "bus"
}

class CoordinateSelector:
    def __init__(self):
        self.coordinates = []
        self.current_frame = None
        self.display_width = 800  # Target display width
        self.original_width = None
        self.original_height = None
        self.scaling_factor = None
        self.display_height = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.coordinates) < 4:
            # Convert display coordinates back to original resolution
            original_x = int(x / self.scaling_factor)
            original_y = int(y / self.scaling_factor)
            
            self.coordinates.append([original_x, original_y])
            
            # Draw point on display frame
            display_frame = self.current_frame.copy()
            for idx, coord in enumerate(self.coordinates):
                display_x = int(coord[0] * self.scaling_factor)
                display_y = int(coord[1] * self.scaling_factor)
                cv2.circle(display_frame, (display_x, display_y), 5, (0, 255, 0), -1)
                cv2.putText(display_frame, str(idx + 1), (display_x + 10, display_y + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            print(f"Point {len(self.coordinates)}: ({original_x}, {original_y})")
            
            if len(self.coordinates) == 4:
                print("\nAll 4 points selected. Press 'q' to continue.")
            
            cv2.imshow('Select 4 Points', display_frame)

    def select_coordinates(self, video_path):
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise ValueError("Error: Could not open video.")

        ret, frame = video.read()
        if not ret:
            raise ValueError("Error: Could not read frame.")

        self.original_height, self.original_width = frame.shape[:2]
        self.scaling_factor = self.display_width / self.original_width
        self.display_height = int(self.original_height * self.scaling_factor)

        self.current_frame = cv2.resize(frame, (self.display_width, self.display_height))

        cv2.namedWindow('Select 4 Points')
        cv2.setMouseCallback('Select 4 Points', self.mouse_callback)

        print("\nSelect 4 points for speed detection zone:")
        print("1. Top left")
        print("2. Top right")
        print("3. Bottom right")
        print("4. Bottom left")

        while True:
            cv2.imshow('Select 4 Points', self.current_frame)
            if cv2.waitKey(1) & 0xFF == ord('q') and len(self.coordinates) == 4:
                break

        video.release()
        cv2.destroyAllWindows()
        return np.array(self.coordinates)

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

def calculate_percentile_speeds(speeds):
    if not speeds:
        return 0, 0, 0
    
    sorted_speeds = sorted(speeds)
    total = len(sorted_speeds)
    
    min_speed_idx = int(0.15 * total)
    safe_speed_idx = int(0.85 * total)
    design_speed_idx = int(0.98 * total)
    
    return (
        sorted_speeds[min_speed_idx] if min_speed_idx < total else 0,
        sorted_speeds[safe_speed_idx] if safe_speed_idx < total else 0,
        sorted_speeds[design_speed_idx] if design_speed_idx < total else 0
    )

def main():
    st.set_page_config(page_title="Vehicle Speed Estimation")
    st.header("Vehicle Speed Estimation")

    source_video_file = st.file_uploader("Upload source video", type=["mp4"])

    if source_video_file is not None:
        # Save the uploaded file to disk
        input_path = "source_video.mp4"
        with open(input_path, "wb") as f:
            f.write(source_video_file.getbuffer())

        # Create target directory if it doesn't exist
        os.makedirs("./target", exist_ok=True)
        
        # Generate output filename based on input
        output_filename = f"{Path(input_path).stem}_out.mp4"
        output_path = os.path.join("./target", output_filename)

        # Get the selected coordinates
        coordinate_selector = CoordinateSelector()
        SOURCE = coordinate_selector.select_coordinates(input_path)
        st.write("Selected coordinates:", SOURCE)

        video_info = sv.VideoInfo.from_video_path(video_path=input_path)
        model = YOLO("best.pt")

        byte_track = sv.ByteTrack(
            frame_rate=video_info.fps, track_activation_threshold=0.3
        )

        thickness = sv.calculate_optimal_line_thickness(
            resolution_wh=video_info.resolution_wh
        )
        text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)
        box_annotator = sv.BoxAnnotator(thickness=thickness)
        label_annotator = sv.LabelAnnotator(
            text_scale=text_scale,
            text_thickness=thickness,
            text_position=sv.Position.BOTTOM_CENTER,
        )
        trace_annotator = sv.TraceAnnotator(
            thickness=thickness,
            trace_length=video_info.fps * 2,
            position=sv.Position.BOTTOM_CENTER,
        )

        frame_generator = sv.get_video_frames_generator(source_path=input_path)
        total_frames = int(cv2.VideoCapture(input_path).get(cv2.CAP_PROP_FRAME_COUNT))

        polygon_zone = sv.PolygonZone(polygon=SOURCE)
        view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

        coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))
        
        # Add a dictionary to store final speeds for each vehicle
        vehicle_speeds = defaultdict(list)

        # Create a video writer to save the output
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        output_video = cv2.VideoWriter(
            output_path,
            fourcc,
            video_info.fps,
            video_info.resolution_wh,
        )

        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for frame_idx, frame in enumerate(frame_generator):
            # Update progress bar
            progress = int((frame_idx + 1) / total_frames * 100)
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {frame_idx + 1}/{total_frames}")
            
            result = model(frame)[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = detections[detections.confidence > 0.3]
            detections = detections[polygon_zone.trigger(detections)]
            detections = detections.with_nms(threshold=0.5)
            detections = byte_track.update_with_detections(detections=detections)

            points = detections.get_anchors_coordinates(
                anchor=sv.Position.BOTTOM_CENTER
            )
            points = view_transformer.transform_points(points=points).astype(int)

            for tracker_id, [_, y] in zip(detections.tracker_id, points):
                coordinates[tracker_id].append(y)

            labels = []
            for tracker_id, class_id in zip(detections.tracker_id, detections.class_id):
                vehicle_type = VEHICLE_CLASSES.get(int(class_id), "unknown")
                
                if len(coordinates[tracker_id]) < video_info.fps / 2:
                    labels.append(f"{vehicle_type}")
                else:
                    coordinate_start = coordinates[tracker_id][-1]
                    coordinate_end = coordinates[tracker_id][0]
                    distance = abs(coordinate_start - coordinate_end)
                    time = len(coordinates[tracker_id]) / video_info.fps
                    speed = distance / time * 3.6
                    vehicle_speeds[tracker_id].append(speed)
                    labels.append(f"{vehicle_type} {int(speed)} km/h")

            # Draw the selected zone on the frame
            zone_points = SOURCE.reshape((-1, 1, 2)).astype(np.int32)
            annotated_frame = frame.copy()
            cv2.polylines(annotated_frame, [zone_points], True, (0, 255, 0), 2)

            annotated_frame = trace_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            annotated_frame = box_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
            )

            # Write the annotated frame to the output video
            output_video.write(annotated_frame)

        # Release the video writer
        output_video.release()
        
        # Calculate final average speeds for each vehicle
        final_speeds = [np.mean(speeds) for speeds in vehicle_speeds.values()]
        
        # Calculate percentile speeds
        min_speed, safe_speed, design_speed = calculate_percentile_speeds(final_speeds)
        
        # Display results
        st.subheader("Speed Analysis Results")
        st.write(f"Total vehicles detected: {len(final_speeds)}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Minimum Speed (15th percentile)", f"{min_speed:.1f} km/h")
        col2.metric("Safe Speed (85th percentile)", f"{safe_speed:.1f} km/h")
        col3.metric("Design Speed (98th percentile)", f"{design_speed:.1f} km/h")
        
        # Clear progress bar and show completion
        status_text.text("Processing complete!")
        
        # Display the output video
        st.video(output_path, format="video/mp4")

if __name__ == "__main__":
    main()