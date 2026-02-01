/**
 * WebRTC Client for Smart Attendance System
 * Handles camera streaming to server via WebRTC
 */

class WebRTCClient {
    constructor(options = {}) {
        this.room = options.room || 'gate';
        this.serverUrl = options.serverUrl || this.getServerUrl();
        this.onConnectionStateChange = options.onConnectionStateChange || (() => { });
        this.onFaceDetections = options.onFaceDetections || (() => { });
        this.onAttendanceMarked = options.onAttendanceMarked || (() => { });
        this.onError = options.onError || console.error;

        this.ws = null;
        this.pc = null;
        this.localStream = null;
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 20; // Increased retry limit
        this.reconnectTimer = null;

        // P2P Configuration
        this.rtcConfig = {
            iceServers: [
                { urls: 'stun:stun.l.google.com:19302' }
            ]
        };
    }

    log(msg) {
        console.log(`[Client] ${msg}`);
        const debug = document.getElementById('debug-console');
        if (debug) {
            const line = document.createElement('div');
            line.textContent = `> ${msg}`;
            debug.appendChild(line);
            debug.scrollTop = debug.scrollHeight;
        }
    }

    getServerUrl() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const url = `${protocol}//${window.location.host}/ws/${this.room}`;
        this.log(`Resolved WS URL: ${url}`);
        return url;
    }

    async start(videoElement) {
        this.videoElement = videoElement;
        this.log("Initializing camera...");

        try {
            // Get camera access
            this.localStream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    frameRate: { ideal: 30 },
                    facingMode: 'user'
                },
                audio: true
            });

            this.log("Camera access granted");

            // Display local video
            videoElement.srcObject = this.localStream;
            videoElement.muted = true;
            await videoElement.play();

            // Connect to signaling server
            await this.connectWebSocket();

            this.onConnectionStateChange('local_stream_ready');

        } catch (error) {
            this.log(`Camera/Start Error: ${error.message}`);
            this.onError('Failed to start camera: ' + error.message);
            throw error;
        }
    }

    async connectWebSocket() {
        if (this.ws) {
            this.ws.close();
        }

        return new Promise((resolve, reject) => {
            const url = this.getServerUrl(); // Refresh URL in case host changes logic?
            this.log(`Connecting to WS: ${url}`);

            try {
                this.ws = new WebSocket(url);
            } catch (e) {
                this.log(`WS Creation Error: ${e.message}`);
                return reject(e);
            }

            this.ws.onopen = () => {
                this.log('WS Connection Opened');
                this.isConnected = true;
                this.reconnectAttempts = 0;
                this.onConnectionStateChange('connected');

                // Keep-alive ping every 10s
                if (this.pingInterval) clearInterval(this.pingInterval);
                this.pingInterval = setInterval(() => this.sendPing(), 10000);

                // Start sending frames for server-side processing
                this.startFrameSending();

                resolve();
            };

            this.ws.onmessage = async (event) => {
                try {
                    const message = JSON.parse(event.data);
                    if (['answer', 'offer', 'viewer_joined'].includes(message.type)) {
                        this.log(`Rx Message: ${message.type}`);
                    }
                    await this.handleMessage(message);
                } catch (e) {
                    this.log(`Msg Error: ${e.message}`);
                    console.error("Error handling message:", e);
                }
            };

            this.ws.onerror = (error) => {
                this.log(`WS Error Event`); // WS errors are often empty in JS
                console.error('WebSocket error:', error);
            };

            this.ws.onclose = (event) => {
                this.log(`WS Closed: Code ${event.code}, Reason: ${event.reason || 'None'}`);
                this.isConnected = false;
                this.onConnectionStateChange('disconnected');
                this.stopFrameSending();
                this.cleanupPeerConnection();
                this.attemptReconnect();
            };
        });
    }

    async handleMessage(message) {
        switch (message.type) {
            case 'viewer_joined':
                this.log('Viewer joined! Starting P2P...');
                await this.createPeerConnection(message.client_id);
                break;

            case 'answer':
                this.log('Received Answer SDP');
                if (this.pc) {
                    await this.pc.setRemoteDescription(new RTCSessionDescription(message.sdp));
                }
                break;

            case 'ice-candidate':
                if (this.pc && message.candidate) {
                    try {
                        await this.pc.addIceCandidate(new RTCIceCandidate(message.candidate));
                    } catch (e) {
                        console.error("Error adding ICE:", e);
                    }
                }
                break;

            case 'face_result':
                // Server detected faces (Known or Unknown)
                if (message.faces) {
                    // Map to 0-1 normalized coordinates based on processed frame size
                    const normalizedFaces = message.faces.map(face => ({
                        name: face.name,
                        status: face.status,
                        // Bbox is [x, y, w, h]
                        x: face.bbox[0] / message.frame_width,
                        y: face.bbox[1] / message.frame_height,
                        width: face.bbox[2] / message.frame_width,
                        height: face.bbox[3] / message.frame_height
                    }));

                    this.onFaceDetections(normalizedFaces);

                    // Trigger sound for known faces only
                    const knownFace = message.faces.find(f => f.status === 'present');
                    if (knownFace) {
                        this.onAttendanceMarked(knownFace);
                    }
                }
                break;
        }
    }

    async createPeerConnection(targetClientId) {
        if (this.pc) this.pc.close();

        this.log("Creating RTCPeerConnection");
        this.pc = new RTCPeerConnection(this.rtcConfig);

        // Add local tracks
        this.localStream.getTracks().forEach(track => {
            this.pc.addTrack(track, this.localStream);
        });

        this.pc.onicecandidate = (event) => {
            if (event.candidate) {
                this.ws.send(JSON.stringify({
                    type: 'ice-candidate',
                    candidate: event.candidate,
                    target_client: targetClientId
                }));
            }
        };

        this.pc.onconnectionstatechange = () => {
            this.log(`P2P State: ${this.pc.connectionState}`);
        };

        // Create Offer
        const offer = await this.pc.createOffer();
        await this.pc.setLocalDescription(offer);

        this.log("Sending Offer");
        this.ws.send(JSON.stringify({
            type: 'offer',
            sdp: offer,
            target_client: targetClientId
        }));
    }

    cleanupPeerConnection() {
        if (this.pc) {
            this.pc.close();
            this.pc = null;
        }
    }

    attemptReconnect() {
        if (this.reconnectTimer) return;

        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            this.onError('Max retries reached. Refresh page.');
            this.log('Max retries reached');
            return;
        }

        this.reconnectAttempts++;
        const delay = 2000;

        this.log(`Reconnecting in ${delay}ms (Attempt ${this.reconnectAttempts})`);
        this.onConnectionStateChange('reconnecting');

        this.reconnectTimer = setTimeout(() => {
            this.reconnectTimer = null;
            this.connectWebSocket().catch(e => {
                this.log(`Retry failed: ${e.message}`);
            });
        }, delay);
    }

    sendPing() {
        if (this.isConnected && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'ping' }));
        }
    }

    stop() {
        if (this.pingInterval) clearInterval(this.pingInterval);
        if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
        this.stopFrameSending();
        if (this.localStream) {
            this.localStream.getTracks().forEach(track => track.stop());
        }
        this.cleanupPeerConnection();
        if (this.ws) this.ws.close();
    }

    startFrameSending() {
        if (this.frameInterval) clearInterval(this.frameInterval);

        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const isClassroom = this.room === 'classroom';

        // Configuration
        // Gate: Fast update (100ms/10fps), Lower Res (320px)
        // Classroom: High Quality (10s), High Res (VGA 640px or better)
        const interval = isClassroom ? 10000 : 66;
        const quality = isClassroom ? 0.9 : 0.6;
        const msgType = isClassroom ? 'classroom_snapshot' : 'process_frame';

        this.frameInterval = setInterval(() => {
            // Check connection status
            if (!this.isConnected || this.ws.readyState !== WebSocket.OPEN) {
                return; // Not connected yet
            }

            // Check video element
            if (!this.videoElement) return;

            // Force play if paused (sometimes browser pauses hidden videos)
            if (this.videoElement.paused) {
                this.videoElement.play().catch(e => console.error("Auto-resume error:", e));
            }

            // Send frame!
            try {
                let width = 640;
                let height = 480;

                if (isClassroom) {
                    // Limit max width to 1280
                    const maxW = 1280;
                    const scale = Math.min(1, maxW / this.videoElement.videoWidth);
                    width = this.videoElement.videoWidth * scale;
                    height = this.videoElement.videoHeight * scale;
                }

                canvas.width = width;
                canvas.height = height;
                ctx.drawImage(this.videoElement, 0, 0, width, height);

                // Quality: lower for gate (speed), higher for classroom
                const jpegQuality = isClassroom ? 0.9 : 0.6;
                const dataUrl = canvas.toDataURL('image/jpeg', jpegQuality);

                this.ws.send(JSON.stringify({
                    type: msgType,
                    frame: dataUrl,
                    timestamp: new Date().toISOString()
                }));
            } catch (e) {
                console.error("Frame send error:", e);
            }
        }, interval);

        this.log(`Started frame sender (${interval}ms)`);
    }


    stopFrameSending() {
        if (this.frameInterval) {
            clearInterval(this.frameInterval);
            this.frameInterval = null;
            this.log("Stopped sending frames");
        }
    }
}

window.WebRTCClient = WebRTCClient;
