"""
Smart Attendance System - Main FastAPI Application

A real-time, offline attendance system using:
- WebRTC for smooth video streaming from mobile cameras
- YuNet for fast face detection
- InsightFace for face recognition
- WebSocket for signaling and real-time updates
"""
import sys
import asyncio
import json
import base64
from datetime import datetime, date
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager
import cv2
import numpy as np

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

import config
from database.database import init_db, get_db, get_session
from database.models import Student, Attendance
from database import mongodb  # MongoDB for face embeddings and analytics
from services.signaling import manager
from services.face_detector import face_detector
from services.face_recognizer import face_recognizer
from services.attendance_service import attendance_service
from services.audio_service import audio_service
from services.behavior_detector import behavior_detector
from services import export
from fastapi.responses import Response, StreamingResponse


# ----- Lifespan Context Manager -----
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup application resources"""
    print("\n" + "="*50)
    print("Smart Attendance System Starting...")
    print("="*50 + "\n")
    
    # Initialize SQLite database (legacy backup)
    await init_db()
    
    # Connect to MongoDB Atlas
    mongo_connected = await mongodb.connect_mongodb()
    if not mongo_connected:
        print("MongoDB connection failed - using SQLite fallback")
    
    # Initialize AI models
    face_detector.initialize()
    face_recognizer.initialize()
    behavior_detector.initialize()  # YOLOv8-pose for behavior
    audio_service.initialize()
    
    print("\n" + "="*50)
    print("System Ready!")
    print(f"Dashboard: http://localhost:{config.PORT}")
    print(f"Gate Camera: http://<your-ip>:{config.PORT}/gate")
    print(f"Classroom Camera: http://<your-ip>:{config.PORT}/classroom")
    print("="*50 + "\n")
    
    yield
    
    # Cleanup
    print("\nShutting down...")
    await mongodb.close_mongodb()
    face_detector.close()
    face_recognizer.close()
    audio_service.close()


# ----- FastAPI App -----
app = FastAPI(
    title="Smart Attendance System",
    description="Offline AI-powered attendance with face recognition",
    version="1.0.0",
    lifespan=lifespan
)

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory=str(config.BASE_DIR / "static")), name="static")

# Templates
templates = Jinja2Templates(directory=str(config.BASE_DIR / "templates"))


# ----- Page Routes -----

@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    """Landing page - Eye-X"""
    return templates.TemplateResponse("landing.html", {"request": request})


@app.get("/project", response_class=HTMLResponse)
async def project_details(request: Request):
    """Deep dive into Project Architecture & Innovations"""
    return templates.TemplateResponse("project_details.html", {"request": request})


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Dashboard - camera feeds only"""
    return templates.TemplateResponse("dashboard_new.html", {"request": request})


@app.get("/add-students", response_class=HTMLResponse)
async def add_students_page(request: Request):
    """Add Students page"""
    return templates.TemplateResponse("add_students.html", {"request": request})


@app.get("/analysis", response_class=HTMLResponse)
async def analysis_page(request: Request):
    """Analytics page with charts"""
    return templates.TemplateResponse("analysis.html", {"request": request})


@app.get("/data", response_class=HTMLResponse)
async def data_page(request: Request):
    """Student data and attendance records"""
    return templates.TemplateResponse("data.html", {"request": request})


@app.get("/gate", response_class=HTMLResponse)
async def gate_camera(request: Request):
    """Gate camera page (for Phone 1)"""
    return templates.TemplateResponse("gate_camera.html", {"request": request})


@app.get("/classroom", response_class=HTMLResponse)
async def classroom_camera(request: Request):
    """Classroom camera page (for Phone 2 - Optional)"""
    return templates.TemplateResponse("classroom_camera.html", {"request": request})


# ----- WebSocket Signaling -----

# Logger Setup
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EYE_X")

@app.websocket("/ws/{room}")
async def websocket_endpoint(websocket: WebSocket, room: str):
    """
    WebSocket endpoint for WebRTC signaling.
    Rooms: 'gate', 'classroom', 'dashboard'
    """
    try:
        connection = await manager.connect(websocket, room)
    except Exception as e:
        logger.error(f"Error connecting to room {room}: {e}")
        import traceback
        traceback.print_exc()
        await websocket.close()
        return

    # Wait for connection to stabilize
    await asyncio.sleep(0.1)
    
    # If a camera just joined and dashboard viewers exist, notify the camera to start P2P
    if room in ["gate", "classroom"]:
        dashboard_connections = manager.get_room_connections("dashboard")
        if dashboard_connections:
            # Notify this camera that a viewer is ready
            for dashboard_conn in dashboard_connections:
                await manager.send_personal(websocket, {
                    "type": "viewer_joined",
                    "client_id": dashboard_conn.client_id
                })
                logger.info(f"Notified {room} camera of existing dashboard viewer")

    try:
        while True:
            try:
                data = await websocket.receive_json()
            except RuntimeError:
                break
            except Exception:
                break
                
            message_type = data.get("type")
            
            # P2P Signaling Relay
            if message_type == "viewer_ready":
                # Dashboard is ready, tell all camera rooms to initiate P2P offer
                for camera_room in ["gate", "classroom"]:
                    await manager.broadcast_to_room(camera_room, {
                        "type": "viewer_joined",
                        "client_id": connection.client_id
                    })
                
            elif message_type == "offer":
                # Camera -> Dashboard (include room name for routing)
                await manager.broadcast_to_room("dashboard", {
                    "type": "offer",
                    "sdp": data["sdp"],
                    "client_id": connection.client_id,
                    "room": room  # gate or classroom
                })
                
            elif message_type == "answer":
                # Dashboard -> Camera (broadcast to all camera rooms)
                for camera_room in ["gate", "classroom"]:
                    await manager.broadcast_to_room(camera_room, {
                        "type": "answer",
                        "sdp": data["sdp"],
                        "target_client": data.get("target_client")
                    })
                
            elif message_type == "ice-candidate":
                # Relay ICE candidates
                if room in ["gate", "classroom"]:
                    # Camera -> Dashboard
                    target_room = "dashboard"
                else:
                    # Dashboard -> All cameras
                    for camera_room in ["gate", "classroom"]:
                        await manager.broadcast_to_room(camera_room, {
                            "type": "ice-candidate",
                            "candidate": data["candidate"],
                            "client_id": connection.client_id
                        })
                    target_room = None
                
                if target_room:
                    await manager.broadcast_to_room(target_room, {
                        "type": "ice-candidate",
                        "candidate": data["candidate"],
                        "client_id": connection.client_id,
                        "from_room": room  # Add this so dashboard knows which camera
                    })
                    
            elif message_type == "process_frame":
                if room == "classroom":
                    await process_behavior(data, websocket)
                else:
                    await process_frame(data, websocket)
            
            elif message_type == "process_behavior":
                # Classroom frame for behavior analysis
                await process_behavior(data, websocket)
            
            elif message_type == "classroom_snapshot":
                # Classroom snapshot for behavior analysis with annotated image
                await process_classroom_snapshot(data, websocket)
                
            elif message_type == "ping":
                await manager.send_personal(websocket, {"type": "pong"})
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {room}")
        await manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error in room {room}: {e}")
        # import traceback
        # traceback.print_exc()
        await manager.disconnect(websocket)


@app.get("/api/students/history")
async def get_students_history():
    """Get students with last 5 days attendance history."""
    return await mongodb.get_students_with_attendance_history(5)


@app.get("/api/analytics/attendance-trend")
async def get_attendance_trend_api(days: int = 7):
    """Get 7-day attendance trend."""
    data = await mongodb.get_attendance_trend(days)
    return {"data": data}

@app.get("/api/analytics/behavior-summary")
async def get_behavior_summary_api():
    """Get today's behavior summary."""
    data = await mongodb.get_behavior_summary()
    return {"data": data}


# ----- Metrics & Export API -----

@app.get("/api/stats/today")
async def get_daily_stats():
    """Get real-time daily statistics for dashboard graphs."""
    attendance_data = await mongodb.get_daily_detailed_attendance()
    return attendance_data

@app.get("/api/behavior/today")
async def get_behavior_stats():
    """Get today's behavior analytics."""
    return await mongodb.get_hourly_behavior()

@app.get("/api/export/daily")
async def download_daily_report(date: str = None):
    """Download daily attendance Excel (.xlsx)."""
    data = await mongodb.get_daily_detailed_attendance(date)
    excel_io = export.generate_daily_excel(data)
    
    filename = f"attendance_daily_{data['date']}.xlsx"
    return Response(
        content=excel_io.getvalue(),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.get("/api/export/monthly")
async def download_monthly_report(month: int = None, year: int = None):
    """Download monthly attendance Excel (.xlsx)."""
    if not month: month = datetime.now().month
    if not year: year = datetime.now().year
    
    data = await mongodb.get_monthly_attendance_grid(year, month)
    excel_io = export.generate_monthly_excel(data)
    
    filename = f"attendance_monthly_{month}_{year}.xlsx"
    return Response(
        content=excel_io.getvalue(),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


async def process_behavior(data: dict, websocket: WebSocket):
    """
    Process frame from classroom camera for behavior detection.
    """
    try:
        # 1. Decode Frame
        frame_data = data.get("frame")
        if not frame_data: return
            
        if "," in frame_data:
            frame_data = frame_data.split(",")[1]
        
        try:
            img_bytes = base64.b64decode(frame_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception:
            return
            
        if frame is None: return

        # 2. Detect Behavior
        detections = await behavior_detector.detect_async(frame)
        
        # 3. Save Stats to MongoDB
        stats = behavior_detector.get_behavior_stats(detections)
        await mongodb.save_behavior_snapshot(stats)

        # 4. Broadcast Results (optional, for overlay)
        # We can draw on frame and send back, or just send stats
        await manager.broadcast_to_room("dashboard", {
            "type": "behavior_update",
            "stats": stats,
            "timestamp": datetime.utcnow().isoformat()
        })
            
    except Exception as e:
        logger.error(f"Error processing behavior: {e}")


async def process_frame(data: dict, websocket: WebSocket):
    """
    Process incoming video frame from gate camera.
    Detect faces, recognize students, and mark attendance.
    """
    try:
        # 1. Decode Frame
        frame_data = data.get("frame")
        if not frame_data:
            return
            
        if "," in frame_data:
            frame_data = frame_data.split(",")[1]
        
        try:
            img_bytes = base64.b64decode(frame_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception:
            return
        
        if frame is None: return

        # 2. Detect Faces
        detections = await face_detector.detect_async(frame)
        if not detections:
            detections = []
        
        processed_faces = []
        
        # 3. Process Detections
        for detection in detections:
            bbox = list(detection.bbox) # [x, y, w, h]
            x, y, w, h = bbox
            
            face_img = face_detector.extract_face(frame, detection)
            if face_img is None: continue
            
            # Recognition
            match = await face_recognizer.recognize_async(face_img)
            
            name = "Unknown"
            color = (0, 0, 255) # Red for unknown
            confidence = 0.0
            status = "unknown"
            
            if match:
                # Known Face
                name = match.name
                color = (0, 255, 0) # Green for known
                confidence = float(match.similarity)
                status = "present"
                
                # Check duplicate before marking
                if not await mongodb.is_already_present_today(match.student_id):
                    # Mark Attendance in MongoDB
                    await mongodb.mark_attendance(
                        student_id=match.student_id,
                        confidence=confidence,
                        auto_detected=True,
                        camera="gate"
                    )
                    
                    # Notify Dashboard
                    await manager.send_personal(websocket, {
                        "type": "new_attendance",
                        "name": name,
                        "roll_number": match.roll_number,
                        "timestamp": datetime.utcnow().isoformat()
                    })
            else:
                # Alert Dashboard for unknown
                await manager.broadcast_to_room("dashboard", {
                    "type": "intrusion_alert",
                    "timestamp": datetime.utcnow().isoformat()
                })

            processed_faces.append({
                "name": name,
                "status": status,
                "bbox": bbox,
                "confidence": confidence
            })

            # Draw bounding box on frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{name} ({int(confidence*100)}%)", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 4. Broadcast Results
        # Send to all cameras and dashboard
        msg = {
            "type": "face_result",
            "faces": processed_faces,
            "frame_width": frame.shape[1],
            "frame_height": frame.shape[0]
        }
        
        for target_room in ["gate", "classroom", "dashboard"]:
            await manager.broadcast_to_room(target_room, msg)
            
    except Exception as e:
        logger.error(f"Error processing frame: {e}")


async def process_behavior(data: dict, websocket: WebSocket):
    """Process incoming video frame from classroom for behavior detection"""
    try:
        frame_data = data.get("frame")
        if not frame_data: return
            
        if "," in frame_data:
            frame_data = frame_data.split(",")[1]
        
        img_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None: return
        
        # Detect behaviors using YOLOv8-pose
        detections = await behavior_detector.detect_async(frame)
        if not detections: return
        
        # Build response with behavior data
        behaviors = []
        frame_height, frame_width = frame.shape[:2]
        
        for det in detections:
            behaviors.append({
                "person_id": det.person_id,
                "behavior": det.behavior,
                "confidence": round(det.confidence, 2),
                "bbox": list(det.bbox)
            })
        
        # Get aggregate stats
        stats = behavior_detector.get_behavior_stats(detections)
        
        # Broadcast to dashboard
        msg = {
            "type": "behavior_result",
            "behaviors": behaviors,
            "stats": stats,
            "frame_width": frame_width,
            "frame_height": frame_height
        }
        await manager.broadcast_to_room("dashboard", msg)
        await manager.broadcast_to_room("classroom", msg)
            
    except Exception as e:
        logger.error(f"Error processing behavior: {e}")


async def process_classroom_snapshot(data: dict, websocket: WebSocket):
    """Process classroom snapshot and return annotated image with both FACE and BEHAVIOR boxes"""
    try:
        frame_data = data.get("frame")
        if not frame_data: return
            
        if "," in frame_data:
            frame_data = frame_data.split(",")[1]
        
        img_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None: return
        
        frame_height, frame_width = frame.shape[:2]
        
        # 1. Run Face Recognition (Sync)
        # First detect faces
        face_detections = face_detector.detect(frame)
        faces = []
        
        for det in face_detections:
            # Crop face for recognition
            x, y, w, h = det.bbox
            x, y = max(0, x), max(0, y)
            w, h = min(w, frame_width - x), min(h, frame_height - y)
            
            if w > 0 and h > 0:
                face_crop = frame[y:y+h, x:x+w]
                match = face_recognizer.recognize(face_crop)
                
                name = match.name if match else "Unknown"
                status = "present" if match else "unknown"
                
                faces.append({
                    "bbox": det.bbox,
                    "name": name,
                    "status": status,
                    # For compatibility with dashboard drawing:
                    "confidence": match.similarity if match else 0.0
                })
        
        # 2. Run Behavior Detection (Async)
        detections = await behavior_detector.detect_async(frame)
        
        # --- Annotation ---
        
        # Draw Faces
        for face in faces:
            x, y, w, h = face['bbox']
            name = face['name']
            
            # Draw green box for faces
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw Behaviors
        colors = {
            'studying': (129, 185, 16),   # green
            'focused': (246, 130, 59),    # blue
            'distracted': (11, 158, 245), # orange
            'sleeping': (68, 68, 239),    # red
            'unknown': (136, 136, 136)    # gray
        }
        
        behaviors_list = []
        for det in detections:
            behaviors_list.append({
                "person_id": det.person_id,
                "behavior": det.behavior,
                "confidence": round(det.confidence, 2)
            })
            
            # Draw behavior box (thicker/different style?)
            # Or just update the existing box? Since Behavior detection is body-based, 
            # the bbox might be larger. We'll draw it too.
            x, y, w, h = det.bbox
            color = colors.get(det.behavior, colors['unknown'])
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            
            # Label
            label = f"{det.behavior} {int(det.confidence * 100)}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, y - 25), (x + tw + 10, y), color, -1)
            cv2.putText(frame, label, (x + 5, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Get aggregate stats
        stats = behavior_detector.get_behavior_stats(detections)
        
        # Encode annotated frame back to base64
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        annotated_image = base64.b64encode(buffer).decode('utf-8')
        
        # Send to dashboard and classroom
        student_count = max(len(faces), len(detections))
        
        msg = {
            "type": "classroom_snapshot",
            "image": annotated_image,
            "student_count": student_count,
            "stats": stats,
            "faces": faces, # Include raw face data
            "behaviors": behaviors_list
        }
        
        await manager.broadcast_to_room("dashboard", msg)
        await manager.broadcast_to_room("classroom", msg)
        
        # Log to MongoDB
        await mongodb.log_classroom_snapshot(msg)
            
    except Exception as e:
        logger.error(f"Error processing classroom snapshot: {e}")

        import traceback
        traceback.print_exc()


# ----- REST API Endpoints -----

@app.get("/api/status")
async def get_status():
    """Get system status"""
    rooms_status = manager.get_all_rooms_status()
    return {
        "status": "online",
        "timestamp": datetime.utcnow().isoformat(),
        "connections": rooms_status,
        "enrolled_students": face_recognizer.get_enrolled_count(),
        "face_detector_ready": face_detector._initialized,
        "face_recognizer_ready": face_recognizer._initialized,
        "behavior_detector_ready": behavior_detector._initialized
    }


@app.post("/api/attendance/manual")
async def manual_attendance(
    student_id: int = Form(...),
    status: str = Form("present"),
    session: AsyncSession = Depends(get_db)
):
    """Manually mark attendance"""
    # Use MongoDB for manual attendance
    try:
        current_student = await mongodb.get_student_by_id(str(student_id)) if isinstance(student_id, int) else None
        # Convert int ID to match standard if needed, or assume manual ID is sent as string form/mongo ID
        # Actually manual attendance form likely sends Mongo ID as string now
        
        # If student_id came as int (legacy), this might fail if we don't look up by legacy ID?
        # But we don't have legacy ID mapping.
        # Let's assume frontend sends string ID now.
        
        result = await mongodb.mark_attendance(
            student_id=str(student_id),
            confidence=1.0,
            auto_detected=False
        )
        
        if not result:
             raise HTTPException(status_code=400, detail="Failed to mark attendance (already marked?)")
             
        message = f"Attendance marked for {student_id}"
        return {"success": True, "message": message}
        
    except Exception as e:
        logger.error(f"Manual attendance error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/attendance/{attendance_id}")
async def update_attendance(
    attendance_id: int,
    status: str = Form(...),
    notes: str = Form(None),
    session: AsyncSession = Depends(get_db)
):
    """Update attendance record"""
    success, message = await attendance_service.update_attendance(
        session, attendance_id, status, notes
    )
    
    if not success:
        raise HTTPException(status_code=400, detail=message)
    
    return {"success": True, "message": message}


@app.delete("/api/attendance/{attendance_id}")
async def delete_attendance(
    attendance_id: int,
    session: AsyncSession = Depends(get_db)
):
    """Delete attendance record"""
    success, message = await attendance_service.delete_attendance(session, attendance_id)
    
    if not success:
        raise HTTPException(status_code=400, detail=message)
    
    return {"success": True, "message": message}


# ----- Student Management -----

@app.get("/api/students")
async def get_students(session: AsyncSession = Depends(get_db)):
    """Get all students"""
    result = await session.execute(
        select(Student).where(Student.is_active == True).order_by(Student.name)
    )
    students = result.scalars().all()
    
    return [{
        "id": s.id,
        "name": s.name,
        "roll_number": s.roll_number,
        "email": s.email,
        "has_face": s.face_embedding is not None,
        "registered_at": s.registered_at.isoformat() if s.registered_at else None
    } for s in students]


@app.post("/api/students")
async def create_student(
    name: str = Form(...),
    roll_number: str = Form(...),
    email: str = Form(None),
    phone: str = Form(None),
    face_image: UploadFile = File(None),
    session: AsyncSession = Depends(get_db)
):
    """Create a new student and enroll their face"""
    try:
        logger.info(f"Creating student: {name}, roll: {roll_number}")
        
        # Check if roll number exists
        result = await session.execute(
            select(Student).where(Student.roll_number == roll_number)
        )
        if result.scalar_one_or_none():
            logger.warning(f"Roll number already exists: {roll_number}")
            raise HTTPException(status_code=400, detail=f"Roll number '{roll_number}' already exists")
        
        # Create student
        student = Student(
            name=name,
            roll_number=roll_number,
            email=email,
            phone=phone
        )
        session.add(student)
        await session.commit()
        await session.refresh(student)
        
        # Enroll face if image provided
        if face_image:
            contents = await face_image.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is not None:
                success, message = face_recognizer.enroll_student(
                    student.id, name, roll_number, image
                )
                
                if success:
                    # Save face image
                    image_path = config.STUDENTS_DIR / f"{student.id}.jpg"
                    cv2.imwrite(str(image_path), image)
                    student.face_image_path = str(image_path)
                    await session.commit()
                else:
                    # Student created but face enrollment failed
                    return {
                        "success": True,
                        "student_id": student.id,
                        "warning": f"Student created but face enrollment failed: {message}"
                    }
        
        return {"success": True, "student_id": student.id, "message": f"Student {name} created"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating student: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating student: {str(e)}")


@app.post("/api/enroll-face-mongodb/{student_id}")
async def enroll_face_mongodb(
    student_id: str,
    face_image: UploadFile = File(...)
):
    """Enroll student's face in MongoDB (supports 3-step)"""
    try:
        # Read image
        contents = await face_image.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        # Get face embedding
        embedding = face_recognizer.get_embedding(image)
        if embedding is None:
            return {"success": False, "message": "No face detected"}
            
        # Save to MongoDB
        # We don't save the image to disk to save space/complexity for now, 
        # but could add it later. Just the embedding is enough for recognition.
        success = await mongodb.add_face_embedding(student_id, embedding)
        
        # Update in-memory recognizer
        # We need to fetch student details again to be safe
        student = await mongodb.get_student_by_id(student_id)
        if student:
            face_recognizer.enroll_student(
                str(student["_id"]), 
                student["name"], 
                student["roll_number"], 
                image # This updates the in-memory embedding
            )
            
        return {"success": True, "message": "Face enrolled"}
        
    except Exception as e:
        logger.error(f"Error enrolling face (MongoDB): {e}")
        return {"success": False, "message": str(e)}


@app.post("/api/enroll-with-files")
async def enroll_with_files(
    name: str = Form(...),
    roll_number: str = Form(...),
    email: str = Form(None),
    phone: str = Form(None),
    images: List[UploadFile] = File(...)
):
    """Enroll student with multiple images (saved to disk, paths in DB)"""
    try:
        # 1. Create Student in MongoDB
        student = await mongodb.create_student(name, roll_number, email, phone)
        student_id = str(student["_id"])
        
        saved_count = 0
        
        # 2. Process Images
        for idx, file in enumerate(images):
            if idx >= 3: break # Limit to 3
            
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is not None:
                # Save to disk
                filename = f"{student_id}_{idx}.jpg"
                path = config.STUDENTS_DIR / filename
                cv2.imwrite(str(path), img)
                
                # Update DB with PATH
                # We reuse add_face_embedding which now stores PATH
                # We need an embedding for the function arg, even if ignored.
                # But wait, face_recognizer.get_embedding calculates it.
                # We should calculate it to Verify quality?
                
                embedding = face_recognizer.get_embedding(img)
                if embedding is not None:
                    # Update DB (Pass Path)
                    await mongodb.add_face_embedding(student_id, embedding, str(path))
                    
                    # Update Memory
                    face_recognizer.enroll_student(
                        student_id, name, roll_number, img
                    )
                    saved_count += 1
        
        if saved_count == 0:
             return JSONResponse(status_code=400, content={"success": False, "message": "No valid faces detected in uploaded images."})

        return {"success": True, "message": f"Student enrolled with {saved_count} images."}

    except Exception as e:
        logger.error(f"Error enrolling with files: {e}")
        return JSONResponse(status_code=500, content={"success": False, "message": str(e)})


@app.post("/api/students/{student_id}/enroll-face")
async def enroll_face(
    student_id: int,
    face_image: UploadFile = File(...),
    session: AsyncSession = Depends(get_db)
):
    """Enroll or update student's face"""
    result = await session.execute(
        select(Student).where(Student.id == student_id)
    )
    student = result.scalar_one_or_none()
    
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    
    contents = await face_image.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    
    success, message = face_recognizer.enroll_student(
        student.id, student.name, student.roll_number, image
    )
    
    if not success:
        raise HTTPException(status_code=400, detail=message)
    
    # Save face image
    image_path = config.STUDENTS_DIR / f"{student.id}.jpg"
    cv2.imwrite(str(image_path), image)
    student.face_image_path = str(image_path)
    await session.commit()
    
    return {"success": True, "message": message}


# Removed enroll_laptop route
# @app.get("/enroll-laptop") ...


@app.delete("/api/students/{student_id}")
async def delete_student(
    student_id: int,
    session: AsyncSession = Depends(get_db)
):
    """Deactivate a student"""
    result = await session.execute(
        select(Student).where(Student.id == student_id)
    )
    student = result.scalar_one_or_none()
    
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    
    student.is_active = False
    face_recognizer.remove_student(student_id)
    await session.commit()
    
    return {"success": True, "message": f"Student {student.name} deactivated"}


# ----- CSV Export -----

@app.get("/api/export/attendance")
async def export_attendance(
    date_str: str = None,
    session: AsyncSession = Depends(get_db)
):
    """Export attendance as CSV"""
    import csv
    import io
    
    target_date = date_str or date.today().isoformat()
    
    result = await session.execute(
        select(Attendance, Student)
        .join(Student)
        .where(Attendance.date == target_date)
        .order_by(Student.roll_number)
    )
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Roll Number', 'Name', 'Entry Time', 'Exit Time', 'Status', 'Confidence'])
    
    for attendance, student in result.all():
        writer.writerow([
            student.roll_number,
            student.name,
            attendance.entry_time.strftime('%H:%M:%S') if attendance.entry_time else '',
            attendance.exit_time.strftime('%H:%M:%S') if attendance.exit_time else '',
            attendance.status,
            f"{attendance.confidence:.2f}"
        ])
    
    output.seek(0)
    
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=attendance_{target_date}.csv"
        }
    )


# ===== MONGODB ANALYTICS API =====

@app.get("/api/analytics/attendance-trend")
async def get_attendance_trend(days: int = 7):
    """Get attendance trend for the last N days."""
    try:
        trend = await mongodb.get_attendance_trend(days)
        return {"success": True, "data": trend}
    except Exception as e:
        logger.error(f"Error getting attendance trend: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/analytics/behavior-summary")
async def get_behavior_summary(date_str: str = None):
    """Get behavior summary for a specific date."""
    try:
        summary = await mongodb.get_behavior_summary(date_str)
        return {"success": True, "data": summary}
    except Exception as e:
        logger.error(f"Error getting behavior summary: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/analytics/hourly-behavior")
async def get_hourly_behavior():
    """Get hourly behavior breakdown for today."""
    try:
        hourly = await mongodb.get_hourly_behavior()
        return {"success": True, "data": hourly}
    except Exception as e:
        logger.error(f"Error getting hourly behavior: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/analytics/weekly")
async def get_weekly_analytics():
    """Get weekly attendance analytics for graphs."""
    try:
        # Get last 7 days from MongoDB
        trend = await mongodb.get_attendance_trend(7)
        # Process logic
        labels = []
        present_data = []
        absent_data = []
        late_data = [] # TODO: Add late tracking
        
        # Sort by date
        trend.sort(key=lambda x: x["date"])
        
        for day in trend:
            # simple day name format
            d = datetime.fromisoformat(day["date"])
            labels.append(d.strftime("%a"))
            present_data.append(day["present"])
            absent_data.append(day["absent"])
            late_data.append(0)
            
        return {
            "labels": labels,
            "present": present_data,
            "absent": absent_data,
            "late": late_data
        }
    except Exception as e:
        logger.error(f"Error fetching weekly analytics: {e}")
        return {"labels": [], "present": [], "absent": [], "late": []}


@app.get("/api/attendance/stats")
async def get_today_stats():
    """Get stats for cards."""
    try:
        today = date.today().isoformat()
        # Re-use trend or get specific
        trend = await mongodb.get_attendance_trend(1) 
        if trend:
            data = trend[0]
            if data["date"] == today:
                 total = data["total"]
                 present = data["present"]
                 pct = (present / total * 100) if total > 0 else 0
                 return {"attendance_percentage": pct}
        return {"attendance_percentage": 0}
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        return {"attendance_percentage": 0}


@app.get("/api/mongodb/students")
async def get_mongodb_students():
    """Get all students from MongoDB."""
    try:
        students = await mongodb.get_all_students()
        # Convert ObjectId to string
        for s in students:
            s["_id"] = str(s["_id"])
            s["has_face"] = len(s.get("face_embeddings", [])) > 0
            s["face_count"] = len(s.get("face_embeddings", []))
        return students
    except Exception as e:
        logger.error(f"Error getting MongoDB students: {e}")
        return []


@app.post("/api/mongodb/students")
async def create_mongodb_student(
    name: str = Form(...),
    roll_number: str = Form(...),
    email: str = Form(None),
    phone: str = Form(None)
):
    """Create a student in MongoDB (without face - for 3-photo enrollment later)."""
    try:
        # Check if exists
        existing = await mongodb.get_student_by_roll(roll_number)
        if existing:
            raise HTTPException(status_code=400, detail=f"Roll number '{roll_number}' already exists")
        
        student = await mongodb.create_student(name, roll_number, email, phone)
        student["_id"] = str(student["_id"])
        
        return {"success": True, "student": student, "message": f"Student {name} created. Now enroll face."}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating MongoDB student: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/mongodb/students/{student_id}/enroll-face")
async def enroll_face_mongodb(
    student_id: str,
    step: int = Form(...),  # 1, 2, or 3
    face_image: str = Form(...)  # Base64 image data
):
    """Enroll a face embedding for 3-photo enrollment."""
    try:
        # Decode image
        if "," in face_image:
            face_image = face_image.split(",")[1]
        
        img_bytes = base64.b64decode(face_image)
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return {"success": False, "error": "Invalid image data"}
        
        # Get face embedding using InsightFace
        embedding = face_recognizer.get_embedding(image)
        
        if embedding is None:
            return {"success": False, "error": "No face detected in image. Please try again."}
        
        # Save image
        image_path = str(config.STUDENTS_DIR / f"{student_id}_face{step}.jpg")
        cv2.imwrite(image_path, image)
        
        # Store in MongoDB
        success = await mongodb.add_face_embedding(student_id, embedding, image_path)
        
        if success:
            return {
                "success": True, 
                "step": step, 
                "message": f"Face {step}/3 enrolled successfully!"
            }
        else:
            return {"success": False, "error": "Failed to save embedding"}
            
    except Exception as e:
        logger.error(f"Error enrolling face: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/attendance/stats")
async def get_attendance_stats():
    """Get today's attendance statistics."""
    try:
        stats = await mongodb.get_attendance_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting attendance stats: {e}")
        return {
            "total_students": 0,
            "present_today": 0,
            "absent_today": 0,
            "attendance_rate": 0
        }


@app.get("/api/attendance/today")
async def get_today_attendance():
    """Get today's attendance records."""
    try:
        records = await mongodb.get_today_attendance()
        # Convert ObjectId and datetime to string
        for r in records:
            r["_id"] = str(r["_id"])
            if "student_id" in r:
                r["student_id"] = str(r["student_id"])
            if "check_in_time" in r and r["check_in_time"]:
                r["entry_time"] = r["check_in_time"].isoformat()
        return records
    except Exception as e:
        logger.error(f"Error getting today's attendance: {e}")
        return []


# ----- Run Server -----

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=config.HOST,
        port=config.PORT,
        reload=True
    )
