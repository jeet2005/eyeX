# Eye-X: Edge-Based Smart Attendance System

## Overview
Eye-X is a distributed attendance and behavioral analytics system designed for educational environments. By leveraging an "Edge-First" architecture, the system performs computationally intensive tasks (face detection and pose estimation) on client devices (smartphones and laptops), significantly reducing server bandwidth requirements and latency.

High-performance WebSocket protocols and Peer-to-Peer (WebRTC) streaming enable real-time monitoring with sub-200ms latency, offering a substantial improvement over traditional RTMP-based solutions.

## Key Features
*   **Distributed Processing**: Facial recognition and behavior logic are decentralized to edge nodes.
*   **Real-Time Analytics**: Instant feedback on class attendance and student engagement levels.
*   **Behavior Classification**: Deterministic state machine classifies behavior into 'Attentive', 'Distracted', or 'Sleeping' based on head pose geometry.
*   **Security**: Implementation of DTLS-SRTP for video encryption and AES-256 for data storage.
*   **Scalability**: Time-series database architecture allows for efficient logging of high-frequency events.

## System Architecture
The system operates on a hybrid Client-Server model:
1.  **Signaling Server**: Manages WebSocket connections and WebRTC handshakes.
2.  **Edge Nodes (Cameras)**: Mobile devices running the capture client. They perform H.264 encoding and initial frame processing.
3.  **Dashboard Client**: The administrative interface that renders video streams and visualizes analytics data.

## Installation

### Prerequisites
*   Python 3.9 or higher
*   pip package manager
*   Virtual environment (recommended)

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

4.  **Download Models**
    The large model files are excluded from the repository to maintain light footprint. Run the setup script to download them:
    ```bash
    python download_models.py
    ```

## Usage

1.  **Start the Application**
    ```bash
    python app.py
    ```

2.  **Access Interfaces**
    *   **Dashboard**: http://localhost:8000/dashboard
    *   **Technical Documentation**: http://localhost:8000/project
    *   **Gate Camera Client**: http://localhost:8000/gate
    *   **Classroom Camera Client**: http://localhost:8000/classroom

## License
This project is licensed under the MIT License. See the LICENSE file for details.
