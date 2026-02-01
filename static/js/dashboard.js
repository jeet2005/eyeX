/**
 * Dashboard JavaScript
 * Handles WebSocket connections, camera feeds, and real-time updates
 */

class Dashboard {
    constructor() {
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.gateConnected = false;
        this.classroomConnected = false;

        this.init();
    }

    init() {
        this.connectWebSocket();
        this.loadStats();
        this.loadAttendance();

        // Refresh data periodically
        setInterval(() => this.loadStats(), 30000);
        setInterval(() => this.loadAttendance(), 10000);

        // Expose for enrollment modal
        window.dashboard = this;
    }

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/dashboard`;

        try {
            this.ws = new WebSocket(wsUrl);

            this.ws.onopen = () => {
                console.log('✅ WebSocket connected');
                this.updateConnectionStatus(true);
                this.reconnectAttempts = 0;
            };

            this.ws.onclose = () => {
                console.log('❌ WebSocket disconnected');
                this.updateConnectionStatus(false);
                this.scheduleReconnect();
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };

            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleMessage(data);
                } catch (e) {
                    console.error('Error parsing message:', e);
                }
            };
        } catch (e) {
            console.error('Failed to connect WebSocket:', e);
            this.scheduleReconnect();
        }
    }

    scheduleReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
            console.log(`Reconnecting in ${delay / 1000}s...`);
            setTimeout(() => this.connectWebSocket(), delay);
        }
    }

    updateConnectionStatus(connected) {
        const statusDot = document.getElementById('ws-status');
        const statusText = document.getElementById('connection-text');

        if (statusDot) {
            statusDot.className = `status-dot ${connected ? 'online' : 'offline'}`;
        }
        if (statusText) {
            statusText.textContent = connected ? 'Connected' : 'Disconnected';
        }
    }

    handleMessage(data) {
        console.log('📨 Received:', data.type);

        switch (data.type) {
            case 'gate_status':
            case 'camera_connected':
                const room = data.room || (data.type === 'gate_status' ? 'gate' : null);
                if (room) this.updateCameraStatus(room, true);
                break;

            case 'camera_disconnected':
                if (data.room) this.updateCameraStatus(data.room, false);
                break;

            case 'gate_snapshot':
                this.handleGateSnapshot(data);
                break;

            case 'classroom_snapshot':
                this.handleClassroomSnapshot(data);
                break;

            case 'attendance_update':
            case 'new_attendance':
                this.loadStats();
                this.loadAttendance();
                this.showToast(`✅ ${data.name} marked present!`, 'success');
                break;

            case 'intrusion_alert':
                this.showToast('⚠️ Unknown person detected at Gate!', 'error');
                const gateFeed = document.getElementById('gate-feed');
                if (gateFeed) {
                    gateFeed.style.border = '4px solid red';
                    setTimeout(() => gateFeed.style.border = 'none', 2000);
                }
                break;

            case 'enrollment_result':
                this.handleEnrollmentResult(data);
                break;

            case 'behavior_update':
            case 'behavior_result':
                if (data.stats) this.updateBehaviorStats(data.stats);
                break;

            case 'face_result':
                // Draw face boxes on the gate video overlay
                const faces = data.faces || [];
                this.drawFaceBoxes(faces, data.frame_width, data.frame_height);

                // Update face count
                const countEl = document.getElementById('gate-face-count');
                if (countEl) {
                    countEl.textContent = `   ${faces.length} Face${faces.length !== 1 ? 's' : ''}`;
                }
                break;
        }
    }

    drawFaceBoxes(faces, frameWidth = 320, frameHeight = 240) {
        const canvas = document.getElementById('gate-overlay');
        const video = document.getElementById('gate-video');
        if (!canvas || !video) return;

        const ctx = canvas.getContext('2d');

        // Match canvas size to video display size
        const rect = video.getBoundingClientRect();
        canvas.width = rect.width;
        canvas.height = rect.height;

        // Clear previous drawings
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        faces.forEach(face => {
            // face.bbox = [x, y, w, h] in frame coordinates
            const [fx, fy, fw, fh] = face.bbox;

            // Scale from frame coords to canvas coords
            const scaleX = canvas.width / frameWidth;
            const scaleY = canvas.height / frameHeight;

            const x = fx * scaleX;
            const y = fy * scaleY;
            const w = fw * scaleX;
            const h = fh * scaleY;

            // Choose color based on status
            const color = face.status === 'present' ? '#10b981' : '#ef4444';

            // Draw bounding box
            ctx.strokeStyle = color;
            ctx.lineWidth = 3;
            ctx.strokeRect(x, y, w, h);

            // Draw label background
            const label = face.name || 'Unknown';
            ctx.font = 'bold 14px Inter, sans-serif';
            const textWidth = ctx.measureText(label).width;
            ctx.fillStyle = color;
            ctx.fillRect(x, y - 22, textWidth + 10, 20);

            // Draw label text
            ctx.fillStyle = 'white';
            ctx.fillText(label, x + 5, y - 7);
        });
    }

    updateCameraStatus(camera, connected) {
        const statusDot = document.getElementById(`${camera}-status`);
        if (statusDot) {
            statusDot.className = `status-dot ${connected ? 'online' : 'offline'}`;
        }

        if (camera === 'gate') {
            this.gateConnected = connected;
        } else {
            this.classroomConnected = connected;
        }
    }

    handleGateSnapshot(data) {
        const container = document.getElementById('gate-feed');
        if (!container) return;

        // Remove placeholder
        const placeholder = container.querySelector('.no-feed');
        if (placeholder) {
            placeholder.style.display = 'none';
        }

        // Show snapshot
        let img = container.querySelector('img');
        if (!img) {
            img = document.createElement('img');
            container.appendChild(img);
        }
        img.src = `data:image/jpeg;base64,${data.image}`;

        this.updateCameraStatus('gate', true);
    }

    handleClassroomSnapshot(data) {
        console.log('🎓 Classroom snapshot received');

        const img = document.getElementById('classroom-snapshot');
        const placeholder = document.getElementById('classroom-placeholder');
        const timestamp = document.getElementById('classroom-timestamp');

        if (img && data.image) {
            img.src = `data:image/jpeg;base64,${data.image}`;
            img.style.display = 'block';

            if (placeholder) {
                placeholder.style.display = 'none';
            }
        }

        if (timestamp) {
            let text = new Date().toLocaleTimeString();
            if (data.student_count !== undefined) {
                text = `👥 ${data.student_count} Students • ${text}`;
            }
            timestamp.textContent = text;
        }

        // Update behavior stats
        if (data.behavior_stats) {
            this.updateBehaviorStats(data.behavior_stats);
        }

        this.updateCameraStatus('classroom', true);
    }

    updateBehaviorStats(stats) {
        if (!stats) return;

        const elements = {
            'stat-studying': `📖 ${stats.studying || 0}`,
            'stat-focused': ` ${stats.focused || 0}`,
            'stat-distracted': `👀 ${stats.distracted || 0}`,
            'stat-sleeping': `😴 ${stats.sleeping || 0}`
        };

        for (const [id, value] of Object.entries(elements)) {
            const el = document.getElementById(id);
            if (el) {
                el.textContent = value;
            }
        }
    }

    handleEnrollmentResult(data) {
        // Handle enrollment capture result
        if (window.enrollmentResolve) {
            if (data.success && data.preview) {
                const previewImg = document.getElementById(`preview-${data.step}`);
                if (previewImg) {
                    previewImg.src = `data:image/jpeg;base64,${data.preview}`;
                    previewImg.style.display = 'block';
                }
            }
            window.enrollmentResolve(data.success);
            window.enrollmentResolve = null;
        }
    }

    async loadStats() {
        try {
            const response = await fetch('/api/attendance/stats');
            const stats = await response.json();

            document.getElementById('total-students').textContent = stats.total_students || 0;
            document.getElementById('present-today').textContent = stats.present_today || 0;
            document.getElementById('absent-today').textContent = stats.absent_today || 0;
            document.getElementById('attendance-rate').textContent = `${stats.attendance_rate || 0}%`;
        } catch (e) {
            console.error('Error loading stats:', e);
        }
    }

    async loadAttendance() {
        try {
            const response = await fetch('/api/attendance/today');
            const records = await response.json();

            const tbody = document.getElementById('attendance-list');
            if (!tbody) return;

            if (records.length === 0) {
                tbody.innerHTML = `
                    <tr>
                        <td colspan="5" style="text-align: center; color: var(--text-secondary);">
                            No attendance records yet
                        </td>
                    </tr>
                `;
                return;
            }

            tbody.innerHTML = records.map(r => `
                <tr>
                    <td>${r.name}</td>
                    <td>${r.roll_number}</td>
                    <td>${new Date(r.entry_time).toLocaleTimeString()}</td>
                    <td><span class="badge badge-success">${r.status}</span></td>
                    <td>${(r.confidence * 100).toFixed(0)}%</td>
                </tr>
            `).join('');
        } catch (e) {
            console.error('Error loading attendance:', e);
        }
    }

    showToast(message, type = 'info') {
        const container = document.getElementById('toast-container');
        if (!container) return;

        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        container.appendChild(toast);

        setTimeout(() => toast.remove(), 3000);
    }
}

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new Dashboard();
});
