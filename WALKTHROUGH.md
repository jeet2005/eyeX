# Eye-X System Walkthrough

This document serves as a comprehensive guide to operating and demonstrating the Eye-X Smart Attendance System.

## 1. System Overview

Eye-X consists of three main components:
1.  **Central Dashboard**: For monitoring and administration.
2.  **Gate Camera Node**: For facial recognition and attendance marking.
3.  **Classroom Camera Node**: For continuous behavioral analysis.

## 2. Operational Workflow

### Phase 1: Environment Configuration
1.  Create a file named `.env` in the project root.
2.  Add your MongoDB connection string:
    ```env
    MONGODB_URI=mongodb+srv://<user>:<password>@cluster.mongodb.net/?retryWrites=true&w=majority
    ```
3.  The application will automatically load this securely.

### Phase 2: Server Startup
Launch the backend server using the command line:
```bash
python app.py
```
You should see the initialization log indicating `Server running on port 8000`.

### Phase 3: Dashboard Access
Open a web browser on the host machine and navigate to:
`http://localhost:8000/dashboard`

**Verification:**
*   Check that the "Total Students" count is displayed.
*   Verify that the "Live Camera Feeds" show the "Signal Lost" status (since cameras are not yet connected).

## 3. Demonstration Script

Follow this script to effectively demonstrate the system capabilities.

### Step A: Connect the Gate Camera
1.  On a mobile device, connect to the same Wi-Fi network as the server.
2.  Open the browser and navigate to `http://<SERVER_IP>:8000/gate`.
3.  Grant camera permissions when prompted.
4.  **Action:** Point the camera at a registered student's face.
5.  **Observation:**
    *   **Mobile Screen:** A green bounding box appears around the face.
    *   **Dashboard:** The "Present" count increments immediately. A notification toast appears: "Student Marked Present".

### Step B: Connect the Classroom Camera
1.  On a second mobile device (or new tab), navigate to `http://<SERVER_IP>:8000/classroom`.
2.  **Action:** Position the camera to view the "class" (a seated individual).
3.  **Observation:**
    *   **Dashboard:** The video feed appears in the "Classroom 101" panel.

### Step C: Demonstrate Behavioral Analytics
Have the subject perform the following actions to test the AI state machine:

1.  **Test "Attentive" State**:
    *   *Action*: Look directly at the camera / screen for 5 seconds.
    *   *Result*: The Dashboard status indicator shows "Attentive" (Green).

2.  **Test "Distracted" State**:
    *   *Action*: Turn head to the left or right (> 35 degrees) for 5 seconds.
    *   *Result*: The Dashboard status indicator changes to "Distracted" (Orange).

3.  **Test "Sleeping" State**:
    *   *Action*: Lower head (`Pitch < 15`) or close eyes.
    *   *Result*: The Dashboard status indicator changes to "Sleeping" (Red).

## 4. Troubleshooting Guide

### Issue: "Signal Lost" on Dashboard
*   **Cause**: The WebRTC handshake failed or the mobile device is on a different network.
*   **Resolution**: Ensure both devices are on the same Subnet (e.g., 192.168.1.x). Disable AP Isolation on the router.

### Issue: Video Lag (> 1 second)
*   **Cause**: Network congestion or low Wi-Fi signal.
*   **Resolution**: move closer to the router. The system will automatically attempt to lower the bitrate.

---
**Eye-X Engineering Team**
