# Eye-X: Edge-Based Smart Attendance System

## 1. Overview
Eye-X is a distributed attendance and behavioral analytics system designed for educational environments. By leveraging an "Edge-First" architecture, the system performs computationally intensive tasks (face detection and pose estimation) on client devices (smartphones and laptops), significantly reducing server bandwidth requirements and latency.

High-performance WebSocket protocols and Peer-to-Peer (WebRTC) streaming enable real-time monitoring with sub-200ms latency, offering a substantial improvement over traditional RTMP-based solutions.

## 2. Key Features
*   **Distributed Processing**: Facial recognition and behavior logic are decentralized to edge nodes, reducing server load by 95%.
*   **Real-Time Analytics**: Instant feedback on class attendance and student engagement levels (Attentive/Distracted/Sleeping).
*   **Deterministic AI**: Uses a mathematical Head Pose state machine (Yaw/Pitch/Roll) rather than "black box" deep learning for behavior classification, ensuring explainability.
*   **Security**: Implementation of DTLS-SRTP for video encryption and AES-256 for data storage.
*   **Scalability**: Time-series database architecture allows for efficient logging of high-frequency events.

## 3. System Architecture
The system operates on a hybrid Client-Server model:
1.  **Signaling Server (Python/FastAPI)**: Manages WebSocket connections and WebRTC handshakes (SDP Exchange).
2.  **Edge Nodes (Cameras)**: Mobile devices (Android/iOS) running the capture client. They perform H.264 encoding and hardware-accelerated stream transmission.
3.  **Dashboard Client**: The administrative interface that renders video streams, overlays AI metadata, and enables export of attendance reports.

## 4. Project Structure
```text
EYE_X/
├── database/            # MongoDB connection and schema definitions
│   └── mongodb.py       # Async motor client wrapper
├── services/            # Core business logic
│   ├── behavior_detector.py  # Head pose estimation logic
│   └── face_recognition.py   # YuNet inference engine
├── static/              # Frontend assets (CSS, JS, Images)
├── templates/           # Jinja2 HTML templates
│   ├── dashboard_new.html    # Main admin console
│   └── project_details.html  # Engineering documentation
├── app.py               # Application entry point (Routes & Sockets)
├── config.py            # Environment configuration
└── requirements.txt     # Python dependencies
```

## 5. Installation & Setup

### Prerequisites
*   Python 3.9 or higher
*   MongoDB instance (Local or Atlas)
*   Webcam (for testing)

### Steps
1.  **Clone the Repository**
    ```bash
    git clone https://github.com/jeet2005/eyeX.git
    cd eyeX
    ```

2.  **Environment Setup**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Linux/MacOS
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download AI Models**
    The system requires the YuNet face detection model.
    ```bash
    python download_models.py
    ```

## 6. Configuration
The application uses a `config.py` file for settings. You can override these using environment variables.

| Variable | Default | Description |
| :--- | :--- | :--- |
| `PORT` | 8000 | The HTTP/WebSocket port. |
| `MONGODB_URL` | mongodb://localhost:27017 | Database connection string. |
| `DEBUG` | True | Enable debug logging. |

## 7. API Reference
The system exposes several REST endpoints for integration:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/stats` | Returns real-time system metrics. |
| `GET` | `/api/export/daily` | Downloads daily attendance CSV. |
| `GET` | `/api/students` | Lists enrolled student metadata. |

## 8. License
This project is licensed under the MIT License. See the LICENSE file for details.

---
© 2026 Eye-X Engineering Team.
