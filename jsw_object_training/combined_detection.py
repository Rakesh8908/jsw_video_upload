import streamlit as st
import cv2
import numpy as np
import time
import math
import requests
import tempfile
import os
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO
from tracker import Tracker

st.set_page_config(
    page_title="JSW Cement Bag Detection",
    page_icon="📦",
    layout="wide"
)

FLASK_API_URL = "http://localhost:5000"

st.title("JSW Cement Bag Detection System")
st.write("Process RTSP camera feeds or upload videos to detect and count cement bags")

@st.cache_resource
def load_models():
    try:
        cement_model = YOLO('F:/jsw20042025/best_cement_bags.pt')
        test_img = np.zeros((640, 640, 3), dtype=np.uint8)
        cement_model(test_img)
    except Exception as e:
        st.warning(f"GPU acceleration not available, falling back to CPU: {str(e)}")
        cement_model = YOLO('F:/jsw20042025/best_cement_bags.pt', device='cpu')
    return cement_model

def load_clusters():
    try:
        response = requests.get(f"{FLASK_API_URL}/clusters")
        clusters = response.json()
        st.session_state.available_clusters = clusters
        return clusters
    except Exception as e:
        st.error(f"Error loading clusters: {str(e)}")
        return []

def send_to_inventory_system(cluster_name, in_count, out_count):
    # Check if we should use mock mode (when API is unavailable)
    use_mock = st.session_state.get('use_mock_api', False)
    
    if use_mock:
        st.info(f"Using mock mode: IN={in_count}, OUT={out_count}, NET={in_count-out_count}")
        return {"status": "success", "message": "Mock inventory update successful"}, 200
    
    # Display the counts being sent to inventory
    st.info(f"Sending to inventory: Cluster={cluster_name}, IN={in_count}, OUT={out_count}, NET={in_count-out_count}")
    
    try:
        # Test connection to API server first
        try:
            st.info(f"Connecting to API at {FLASK_API_URL}...")
            response = requests.get(f"{FLASK_API_URL}/clusters", timeout=5)
            response.raise_for_status()  # Raise exception for 4XX/5XX responses
            clusters = response.json()
            st.success(f"Successfully connected to API. Found {len(clusters)} clusters.")
        except requests.exceptions.RequestException as conn_err:
            st.error(f"Cannot connect to inventory API at {FLASK_API_URL}: {str(conn_err)}")
            st.info("Enabling mock mode for future updates. Restart the application to try connecting to the API again.")
            st.session_state.use_mock_api = True
            return {"error": f"API connection failed: {str(conn_err)}"}, 500

        # Find if cluster exists
        cluster_exists = False
        cluster_id = None

        for cluster in clusters:
            if cluster['name'] == cluster_name:
                cluster_exists = True
                cluster_id = cluster['id']
                st.info(f"Found existing cluster: {cluster_name} (ID: {cluster_id})")
                break

        # Process based on whether cluster exists
        if cluster_exists:
            movement_success = True
            error_details = []

            # Process IN movement if needed
            if in_count > 0:
                st.info(f"Recording IN movement of {in_count} bags for cluster {cluster_name}")
                in_data = {"movement_type": "IN", "quantity": in_count}
                try:
                    in_response = requests.post(
                        f"{FLASK_API_URL}/clusters/{cluster_id}/movement", 
                        json=in_data,
                        timeout=5
                    )
                    
                    # Check response and log details
                    if in_response.status_code in (200, 201):
                        st.success(f"IN movement recorded successfully")
                    else:
                        st.error(f"IN movement failed with status code: {in_response.status_code}")
                        st.error(f"Response: {in_response.text}")
                        movement_success = False
                        error_details.append(f"IN movement error: Status {in_response.status_code}")
                        
                except requests.exceptions.RequestException as e:
                    movement_success = False
                    error_details.append(f"IN movement error: {str(e)}")
                    st.error(f"Exception during IN movement: {str(e)}")

            # Process OUT movement if needed
            if out_count > 0:
                st.info(f"Recording OUT movement of {out_count} bags for cluster {cluster_name}")
                out_data = {"movement_type": "OUT", "quantity": out_count}
                try:
                    out_response = requests.post(
                        f"{FLASK_API_URL}/clusters/{cluster_id}/movement", 
                        json=out_data,
                        timeout=5
                    )
                    
                    # Check response and log details
                    if out_response.status_code in (200, 201):
                        st.success(f"OUT movement recorded successfully")
                    else:
                        st.error(f"OUT movement failed with status code: {out_response.status_code}")
                        st.error(f"Response: {out_response.text}")
                        movement_success = False
                        error_details.append(f"OUT movement error: Status {out_response.status_code}")
                        
                except requests.exceptions.RequestException as e:
                    movement_success = False
                    error_details.append(f"OUT movement error: {str(e)}")
                    st.error(f"Exception during OUT movement: {str(e)}")

            # Return result based on success
            if movement_success:
                st.success("All inventory movements recorded successfully")
                return {"status": "success", "message": "Movement records created successfully"}, 200
            else:
                error_msg = "; ".join(error_details)
                st.error(f"Movement API errors: {error_msg}")
                return {"error": error_msg}, 500
        else:
            # Create new cluster with net count
            net_count = in_count - out_count
            st.info(f"Creating new cluster '{cluster_name}' with initial bag count {net_count}")
            data = {"name": cluster_name, "bag_count": net_count}
            try:
                response = requests.post(f"{FLASK_API_URL}/clusters", json=data, timeout=5)
                
                # Check response and log details
                if response.status_code in (200, 201):
                    st.success(f"New cluster created successfully")
                    try:
                        return response.json(), 201
                    except ValueError:
                        st.warning("API returned success but no valid JSON response")
                        return {"status": "success", "message": "Cluster created but no data returned"}, 201
                else:
                    st.error(f"Cluster creation failed with status code: {response.status_code}")
                    st.error(f"Response: {response.text}")
                    return {"error": f"Cluster creation failed with status {response.status_code}"}, 500
                    
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to create new cluster: {str(e)}")
                return {"error": f"Cluster creation failed: {str(e)}"}, 500
    except Exception as e:
        st.error(f"Unexpected error in inventory system: {str(e)}")
        return {"error": str(e)}, 500

LINE_COLORS = [
    (0, 255, 0),
    (0, 0, 255),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128)
]

def main():
    cement_model = load_models()

    st.sidebar.header("Detection Mode")
    mode = st.sidebar.radio("Select Mode", ("RTSP Camera", "Upload Video"))
    
    # Initialize mock API mode if not already set
    if 'use_mock_api' not in st.session_state:
        st.session_state.use_mock_api = False
    
    # Option to toggle mock mode
    mock_mode = st.sidebar.checkbox("Use Mock API (when inventory API unavailable)", 
                                   value=st.session_state.use_mock_api)
    st.session_state.use_mock_api = mock_mode

    st.sidebar.header("Cluster")
    clusters = load_clusters()
    cluster_names = [c['name'] for c in clusters] if clusters else []
    selected_cluster = st.sidebar.selectbox("Select Cluster", cluster_names) if cluster_names else None

    if not selected_cluster:
        st.warning("No cluster selected or available. Please add one via backend.")
        return

    # Allow manual input of a single crossing line
    st.sidebar.header("Crossing Line")
    line_start = st.sidebar.text_input("Line Start (x1, y1)", "100, 300")
    line_end = st.sidebar.text_input("Line End (x2, y2)", "500, 300")

    # Parse input into coordinates
    crossing_line = [tuple(map(int, line_start.split(','))), tuple(map(int, line_end.split(',')))]    
    total_in, total_out = 0, 0

    # Confidence threshold slider
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3)

    tracker = Tracker()
    
    # Dictionary to store previous positions of objects
    previous_positions = {}
    
    # Set to track objects that have crossed the line
    crossed_objects = set()

    def is_crossing_line(point, line):
        """Check if a point is on the line side"""
        x, y = point
        (x1, y1), (x2, y2) = line
        
        # Calculate line equation: ax + by + c = 0
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2
        
        # Calculate the value of the line equation at the point
        value = a * x + b * y + c
        
        # Return the sign of the value (positive or negative)
        return 1 if value > 0 else -1

    if mode == "Upload Video":
        uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
        if uploaded_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())

            cap = cv2.VideoCapture(tfile.name)

            stframe = st.empty()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = cement_model(frame, conf=confidence_threshold)[0]
                boxes = []
                for r in results.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = r
                    if score > confidence_threshold:
                        boxes.append([int(x1), int(y1), int(x2)-int(x1), int(y2)-int(y1)])
                bbox_idx = tracker.update(boxes)
                for x, y, w, h, id in bbox_idx:
                    # Calculate center of the bounding box
                    center_x = x + w // 2
                    center_y = y + h // 2
                    center = (center_x, center_y)
                    
                    # Check if we have seen this object before
                    if id in previous_positions:
                        prev_side = is_crossing_line(previous_positions[id], crossing_line)
                        current_side = is_crossing_line(center, crossing_line)
                        
                        # Check if the object has crossed the line
                        if prev_side != current_side and id not in crossed_objects:
                            if prev_side < 0 and current_side > 0:
                                total_in += 1
                            else:
                                total_out += 1
                            crossed_objects.add(id)
                    
                    # Update the previous position
                    previous_positions[id] = center
                    
                    # Draw bounding box and ID
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID:{id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw the crossing line
                cv2.line(frame, crossing_line[0], crossing_line[1], (255, 0, 0), 2)
                
                # Display counts
                cv2.putText(frame, f"IN: {total_in}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"OUT: {total_out}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(frame, f"NET: {total_in - total_out}", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                
                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            cap.release()

            st.success("Video processing complete.")
            response, status = send_to_inventory_system(selected_cluster, total_in, total_out)
            if status == 200:
                st.success("Inventory updated successfully.")
            else:
                st.error("Failed to update inventory.")

    elif mode == "RTSP Camera":
        rtsp_url = st.text_input("Enter RTSP URL", "")
        if st.button("Start Stream") and rtsp_url:
            cap = cv2.VideoCapture(rtsp_url)

            stframe = st.empty()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = cement_model(frame, conf=confidence_threshold)[0]
                boxes = []
                for r in results.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = r
                    if score > confidence_threshold:
                        boxes.append([int(x1), int(y1), int(x2)-int(x1), int(y2)-int(y1)])
                bbox_idx = tracker.update(boxes)
                for x, y, w, h, id in bbox_idx:
                    # Calculate center of the bounding box
                    center_x = x + w // 2
                    center_y = y + h // 2
                    center = (center_x, center_y)
                    
                    # Check if we have seen this object before
                    if id in previous_positions:
                        prev_side = is_crossing_line(previous_positions[id], crossing_line)
                        current_side = is_crossing_line(center, crossing_line)
                        
                        # Check if the object has crossed the line
                        if prev_side != current_side and id not in crossed_objects:
                            if prev_side < 0 and current_side > 0:
                                total_in += 1
                            else:
                                total_out += 1
                            crossed_objects.add(id)
                    
                    # Update the previous position
                    previous_positions[id] = center
                    
                    # Draw bounding box and ID
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID:{id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw the crossing line
                cv2.line(frame, crossing_line[0], crossing_line[1], (255, 0, 0), 2)
                
                # Display counts
                cv2.putText(frame, f"IN: {total_in}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"OUT: {total_out}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(frame, f"NET: {total_in - total_out}", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                
                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            cap.release()

if __name__ == "__main__":
    main()
