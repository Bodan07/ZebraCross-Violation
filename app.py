import os
import cv2
import streamlit as st
from ultralytics import YOLO
from datetime import datetime
import shutil
import time

def load_model():
    """Load the models for zebra cross segmentation and vehicle detection."""
    model_zebra_cross = YOLO("E:\\mil TA\\weight_v2\\811_yolov9e_100ep_001lr_AdamW.pt")
    model_kendaraan = YOLO("yolov9e.pt")
    return model_zebra_cross, model_kendaraan

def get_largest_zebra_cross_mask(video_path, model_zebra_cross, output_video_path="predicted_output.mp4"):
    """Get the largest zebra cross mask from the first 10 seconds of the video and save the frames as a video."""
    save_dir = "runs/segment/"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)  # Clear the directory to avoid mixing results
    os.makedirs(save_dir, exist_ok=True)
    
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)  # Frames per second
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))  # Frame width
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Frame height
    max_frames = int(fps * 10)  # Process only the first 10 seconds

    # Initialize video writer to save output video
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Codec for MP4
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    largest_mask = None
    largest_area = 0
    frame_count = 0

    while video.isOpened() and frame_count < max_frames:
        ret, frame = video.read()
        if not ret:
            break

        # Resize frame for YOLO processing
        frame_resized = cv2.resize(frame, (640, 384))
        results = model_zebra_cross.predict(frame_resized, imgsz=640, conf=0.6, device=0, save=True, verbose=False,save_dir=save_dir)

        # Extract masks and determine the largest zebra cross mask
        for result in results:
            if hasattr(result, 'masks') and result.masks is not None:
                for mask_data in result.masks.data:
                    mask = mask_data.cpu().numpy().squeeze()
                    mask_area = mask.sum()

                    if mask_area > largest_area:
                        largest_mask = mask
                        largest_area = mask_area

        # Draw results directly on the frame
        annotated_frame = results[0].plot()  # Get the annotated frame from YOLO
        annotated_frame = cv2.resize(annotated_frame, (width, height))  # Resize back to original size

        # Write the annotated frame to the output video
        video_writer.write(annotated_frame)

        frame_count += 1

    video.release()
    video_writer.release()  # Finalize the output video
    print(f"Predicted frames saved to video: {output_video_path}")
    return largest_mask


def detect_vehicles_in_zebra_cross(video_path, model_kendaraan, largest_mask, start_frame, output_video_path="output_detected.mp4"):
    """Detect if any vehicle violates the zebra cross starting from a given frame and save the frames as a video."""
    save_dir = "runs/detect/"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)  # Clear the directory to avoid mixing results
    os.makedirs(save_dir, exist_ok=True)

    # Video Capture
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize video writer to save output video
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Codec for MP4
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    violation_detected = False
    min_violation_time = 5
    min_violation_frames = min_violation_time * fps
    tracked_vehicles = {}

    # Skip to the specified start frame
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        # Resize frame for YOLO processing
        frame_resized = cv2.resize(frame, (640, 384))
        results = model_kendaraan.predict(
            frame_resized, imgsz=640, conf=0.3, classes=[2, 3, 5, 7], save=True, save_dir=save_dir, iou=0.9, device=0, verbose=False
        )

        current_frame_number = int(video.get(cv2.CAP_PROP_POS_FRAMES))

        # Annotate results directly onto the frame
        annotated_frame = results[0].plot()
        annotated_frame = cv2.resize(annotated_frame, (width, height))  # Resize back to original frame size

        for result in results:
            for bbox in result.boxes:
                vehicle_bbox = bbox.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, vehicle_bbox)

                # Ensure bounding box is within the mask dimensions
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(largest_mask.shape[1], x2), min(largest_mask.shape[0], y2)

                # Calculate overlap
                vehicle_mask_area = largest_mask[y1:y2, x1:x2]
                overlap_count = vehicle_mask_area.sum()
                total_vehicle_area = (x2 - x1) * (y2 - y1)

                if total_vehicle_area > 0:
                    overlap_percentage = overlap_count / total_vehicle_area
                else:
                    overlap_percentage = 0

                vehicle_id = (x1, y1, x2, y2)

                if overlap_percentage > 0.1:
                    if vehicle_id not in tracked_vehicles:
                        tracked_vehicles[vehicle_id] = {
                            "start_frame": current_frame_number,
                            "last_frame": current_frame_number,
                            "violation_saved": False
                        }
                    tracked_vehicles[vehicle_id]["last_frame"] = current_frame_number

                    # Check if violation duration exceeds minimum time and hasn't been saved
                    if (current_frame_number - tracked_vehicles[vehicle_id]["start_frame"] >= min_violation_frames and
                            not tracked_vehicles[vehicle_id]["violation_saved"]):
                        violation_detected = True

                        # Save the violation image and mark it as saved
                        save_violation_image(frame, vehicle_id, video_path)
                        tracked_vehicles[vehicle_id]["violation_saved"] = True  # Mark as saved
                        break

            if violation_detected:
                break

        # Write the annotated frame to the output video
        video_writer.write(annotated_frame)

    video.release()
    video_writer.release()  # Finalize the output video
    print(f"Detected frames saved to video: {output_video_path}")
    return violation_detected


def save_violation_image(frame, vehicle_id, video_path):
    """Save the violation image with bounding box."""
    x1, y1, x2, y2 = vehicle_id

    # Dapatkan dimensi asli frame
    height_original, width_original = frame.shape[:2]
    height_resized, width_resized = 384, 640  # Ukuran resize untuk YOLO

    # Ubah bounding box ke dimensi asli
    x1 = int(x1 * width_original / width_resized)
    y1 = int(y1 * height_original / height_resized)
    x2 = int(x2 * width_original / width_resized)
    y2 = int(y2 * height_original / height_resized)

    # Gambarkan bounding box pada frame asli
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Simpan gambar pelanggaran
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    directory_path = "pelanggaran"
    os.makedirs(directory_path, exist_ok=True)

    date_now = datetime.now().strftime("%d_%m_%Y_%H-%M-%S")
    violation_image_path = f"{directory_path}/pelanggaran_{date_now}.jpg"
    cv2.imwrite(violation_image_path, frame)


def display_all_violation_images():
    """Display all violation images in a single panel, changing every few seconds."""
    violation_directory = "pelanggaran"
    violation_images = get_violation_images(violation_directory)

    if violation_images:
        # Create an empty placeholder to display the images
        image_placeholder = st.empty()

        # Loop through all images and display them one by one
        for image_path in violation_images:
            # Display the current image
            image_placeholder.image(image_path, caption="Violation Detected", use_container_width=True)
            # Wait for 3 seconds before changing to the next image
            time.sleep(0.5)  # Adjust the time delay as needed
    else:
        st.write("No violation images found.")


def get_violation_images(directory):
    """Get all violation images from the directory and sort them by creation time."""
    files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    if not files:
        return []
    # Sort images by creation time (oldest to newest)
    files.sort(key=os.path.getctime)
    return files


def main():
    st.title("YOLO Zebra Cross Violation Detection")

    st.subheader("Upload Video")
    uploaded_video = st.file_uploader("Upload a video file for violation detection", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_video.read())

        st.video("temp_video.mp4")
        
        if st.button("Start Detection"):
            # Clear 'pelanggaran' folder before starting detection
            violation_directory = "pelanggaran"
            if os.path.exists(violation_directory):
                for filename in os.listdir(violation_directory):
                    file_path = os.path.join(violation_directory, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
            
            model_zebra_cross, model_kendaraan = load_model()
            video_path = "temp_video.mp4"

            # Create two columns for side-by-side display
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Zebra Cross Segmentation")
                zebra_status = st.text("On progress")

            with col2:
                st.subheader("Vehicle Detection")
                vehicle_status = st.text("On progress")

            # Proses segmentasi dan buat video
            output_zebra_video = "output_zebra_cross.mp4"
            largest_mask = get_largest_zebra_cross_mask(video_path, model_zebra_cross, output_zebra_video)

            # Update tampilan setelah segmentasi selesai
            if largest_mask is None:
                zebra_status.text("No zebra cross detected.")
                st.error("No zebra cross detected in the video.")
                return
            else:
                zebra_status.text("")
                with col1:
                    st.video(output_zebra_video)

            # Proses prediksi kendaraan dan buat video
            fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)
            start_frame = int(fps * 10)
            output_vehicle_video = "output_vehicle.mp4"
            violation_detected = detect_vehicles_in_zebra_cross(video_path, model_kendaraan, largest_mask, start_frame, output_vehicle_video)

            # Update tampilan setelah deteksi kendaraan selesai
            vehicle_status.text("")
            with col2:
                st.video(output_vehicle_video)

            # Menampilkan status pelanggaran
            if violation_detected:
                st.success("Violation detected.")
            else:
                st.info("No violations detected.")

            st.subheader("Violation Detected")
            display_all_violation_images()


if __name__ == "__main__":
    main()

