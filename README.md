# JSW Cement Bag Detection System

A comprehensive computer vision solution for tracking and counting cement bags in industrial settings. The system integrates RTSP camera feeds, YOLOv8 object detection, Flask backend API, Streamlit detection application, and React frontend dashboard for real-time monitoring and inventory management.

## System Components

1. **Flask Backend API**: Database and REST API server for inventory management
2. **Streamlit Detection Application**: YOLOv8-based cement bag detection with both RTSP camera and video upload support
3. **React Frontend Dashboard**: User-friendly interface for monitoring inventory and bag movements
4. **SQLite/MySQL Database**: Storage for clusters and bag movement records

## Setup Instructions

### Prerequisites

- Python 3.10+
- Node.js and npm
- YOLOv8 model file (`best_cement_bags.pt`)

### Python Dependencies

```bash
pip install flask flask-cors flask-sqlalchemy sqlalchemy pymysql ultralytics opencv-python numpy streamlit
```

## Running the Complete System

### Step 1: Start the Flask Backend API

```bash
# Navigate to the backend folder
cd F:/jsw20042025/backend/inference

# Run the SQLite version of the server
python server_sqlite.py

# Alternatively, if MySQL is configured:
# python server.py
```

The Flask API will be available at http://localhost:5000

### Step 2: Start the Streamlit Detection Application

```bash
# Navigate to the detection application folder
cd F:/jsw20042025/jsw_object_training

# Run the combined detection application
streamlit run combined_detection.py
```

The detection interface will be available at http://localhost:8501

### Step 3: Start the React Frontend Dashboard

```bash
# Navigate to the frontend folder
cd F:/jsw20042025/frontend

# Install dependencies (first time only)
npm install

# Run the development server
npm start
```

The dashboard will be available at http://localhost:3000

## Using the System

### 1. Streamlit Detection Application

#### Setup Process

1. **Cluster Management**: First, select or create a cluster for inventory tracking in the "Cluster Management" tab
2. **Line Configuration**: Configure detection lines in the "Line Configuration" tab
3. **Detection Mode**: Choose between RTSP camera feed or video upload in the "Detection Mode" tab

#### RTSP Camera Mode

1. Enter the RTSP base URL (default: `rtsp://admin:Fidelis12@103.21.79.245:554/Streaming/Channels/`)
2. Enter the channel number (101-701)
3. Click "Start RTSP Stream" to begin processing
4. The system will automatically update inventory counts in the database

#### Video Upload Mode

1. Upload a video file (MP4, AVI, MOV formats supported)
2. Click "Process Video" to analyze
3. Results will be displayed with IN/OUT counts
4. The inventory system will be updated accordingly

### 2. React Frontend Dashboard

- **Dashboard**: View overall statistics, cluster utilization, and daily movement summary
- **Cluster Management**: Create, edit, delete, and view clusters and their inventory
- **Camera Feed**: Monitor real-time camera feeds when available

## System Architecture

### Detection System
- **YOLOv8**: Object detection for cement bags
- **Custom Tracking**: Assigns IDs to bags for consistent tracking
- **Line Crossing Detection**: Identifies bag movements across defined lines

### Backend API
- **Flask**: Lightweight web framework for the REST API
- **SQLAlchemy**: ORM for database operations
- **SQLite/MySQL**: Database for storing cluster and movement data

### Frontend
- **React**: JavaScript library for building the user interface
- **Material-UI**: Component library for modern UI elements
- **Recharts**: Charting library for data visualization

## API Endpoints

### Clusters
- `GET /clusters`: List all clusters
- `POST /clusters`: Create a new cluster
- `GET /clusters/<id>`: Get a specific cluster
- `PUT /clusters/<id>`: Update a cluster
- `DELETE /clusters/<id>`: Delete a cluster

### Movements
- `POST /clusters/<id>/movement`: Record bag movement for a cluster
- `GET /clusters/<id>/movements`: Get movement history for a cluster
- `GET /clusters/daily-summary`: Get daily movement summary for all clusters

### Utilities
- `POST /clusters/<id>/reset`: Reset a cluster's count and history
- `GET /`: API information and available endpoints

## Troubleshooting

### Database Connection Issues
- For MySQL connection issues, ensure MySQL server is running
- If MySQL connection fails, use the SQLite version (server_sqlite.py)

### RTSP Connection Issues
- Verify network connectivity to the cameras
- Ensure correct RTSP URL format: `rtsp://admin:Fidelis12@103.21.79.245:554/Streaming/Channels/101`
- Check camera credentials in the URL

### Frontend API Issues
- Ensure the Flask backend is running on port 5000
- Check browser console for specific API errors

## Credits

Developed by Fidelisgroupdev - JSW Backend Team
