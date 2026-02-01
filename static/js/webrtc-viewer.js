/**
 * WebRTC Viewer for Dashboard
 * Receives live video streams from camera clients via WebRTC
 */

class WebRTCViewer {
    constructor(options = {}) {
        this.serverUrl = this.getServerUrl();
        this.onStreamReceived = options.onStreamReceived || (() => { });
        this.onConnectionStateChange = options.onConnectionStateChange || (() => { });

        this.ws = null;
        this.peerConnections = {}; // room -> RTCPeerConnection
        this.remoteStreams = {}; // room -> MediaStream
        this.isConnected = false;

        // ICE servers for NAT traversal
        this.rtcConfig = {
            iceServers: [
                { urls: 'stun:stun.l.google.com:19302' },
                { urls: 'stun:stun1.l.google.com:19302' }
            ]
        };
    }

    getServerUrl() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        return `${protocol}//${window.location.host}/ws/dashboard`;
    }

    async start() {
        console.log('[Viewer] Starting WebRTC viewer...');
        await this.connectWebSocket();
    }

    async connectWebSocket() {
        return new Promise((resolve, reject) => {
            console.log('[Viewer] Connecting to:', this.serverUrl);

            try {
                this.ws = new WebSocket(this.serverUrl);
            } catch (e) {
                console.error('[Viewer] WebSocket creation error:', e);
                return reject(e);
            }

            this.ws.onopen = () => {
                console.log('[Viewer] WebSocket connected');
                this.isConnected = true;
                this.onConnectionStateChange('connected');

                // Tell server we're ready to receive video
                this.ws.send(JSON.stringify({ type: 'viewer_ready' }));
                console.log('[Viewer] Sent viewer_ready');

                resolve();
            };

            this.ws.onmessage = async (event) => {
                try {
                    const message = JSON.parse(event.data);
                    await this.handleMessage(message);
                } catch (e) {
                    console.error('[Viewer] Message error:', e);
                }
            };

            this.ws.onerror = (error) => {
                console.error('[Viewer] WebSocket error:', error);
            };

            this.ws.onclose = () => {
                console.log('[Viewer] WebSocket closed');
                this.isConnected = false;
                this.onConnectionStateChange('disconnected');

                // Cleanup all peer connections
                this.cleanupAllConnections();

                // Attempt reconnect after 3 seconds
                setTimeout(() => this.connectWebSocket(), 3000);
            };
        });
    }

    async handleMessage(message) {
        console.log('[Viewer] Received:', message.type);

        switch (message.type) {
            case 'offer':
                // Camera is offering to stream video
                console.log(`[Viewer] Received offer from ${message.room || 'unknown'}`);
                await this.handleOffer(message);
                break;

            case 'ice-candidate':
                // ICE candidate from camera
                await this.handleIceCandidate(message);
                break;

            case 'camera_connected':
                console.log(`[Viewer] Camera connected: ${message.room}`);
                this.onConnectionStateChange('camera_connected', message.room);
                break;

            case 'camera_disconnected':
                console.log(`[Viewer] Camera disconnected: ${message.room}`);
                this.cleanupConnection(message.room);
                this.onConnectionStateChange('camera_disconnected', message.room);
                break;

            // Pass through other messages (gate_snapshot, face_result, etc.)
            default:
                // These are handled by dashboard.js
                break;
        }
    }

    async handleOffer(message) {
        const room = message.room || 'gate';
        const clientId = message.client_id;

        console.log(`[Viewer] Processing offer for room: ${room}`);

        // Create peer connection for this camera
        const pc = new RTCPeerConnection(this.rtcConfig);
        this.peerConnections[room] = pc;

        // Handle incoming tracks (video/audio)
        pc.ontrack = (event) => {
            console.log(`[Viewer] Received track from ${room}:`, event.track.kind);

            let stream = event.streams[0];
            if (!stream) {
                console.log(`[Viewer] No stream in track event, creating new MediaStream`);
                stream = new MediaStream([event.track]);
            }

            this.remoteStreams[room] = stream;
            this.onStreamReceived(room, stream);
        };

        // Handle ICE candidates
        pc.onicecandidate = (event) => {
            if (event.candidate) {
                this.ws.send(JSON.stringify({
                    type: 'ice-candidate',
                    candidate: event.candidate,
                    target_client: clientId
                }));
            }
        };

        // Connection state monitoring
        pc.onconnectionstatechange = () => {
            console.log(`[Viewer] P2P state for ${room}: ${pc.connectionState}`);
            if (pc.connectionState === 'connected') {
                this.onConnectionStateChange('p2p_connected', room);
            } else if (pc.connectionState === 'failed' || pc.connectionState === 'disconnected') {
                this.onConnectionStateChange('p2p_disconnected', room);
            }
        };

        try {
            // Set remote description (the offer)
            await pc.setRemoteDescription(new RTCSessionDescription(message.sdp));
            console.log(`[Viewer] Set remote description for ${room}`);

            // Create and send answer
            const answer = await pc.createAnswer();
            await pc.setLocalDescription(answer);

            console.log(`[Viewer] Sending answer to ${room}`);
            this.ws.send(JSON.stringify({
                type: 'answer',
                sdp: answer,
                target_client: clientId
            }));
        } catch (e) {
            console.error(`[Viewer] Error handling offer for ${room}:`, e);
        }
    }

    async handleIceCandidate(message) {
        const room = message.from_room || 'gate';
        const pc = this.peerConnections[room];

        if (pc && message.candidate) {
            try {
                await pc.addIceCandidate(new RTCIceCandidate(message.candidate));
            } catch (e) {
                console.error(`[Viewer] Error adding ICE candidate for ${room}:`, e);
            }
        }
    }

    cleanupConnection(room) {
        const pc = this.peerConnections[room];
        if (pc) {
            pc.close();
            delete this.peerConnections[room];
        }
        delete this.remoteStreams[room];
    }

    cleanupAllConnections() {
        for (const room of Object.keys(this.peerConnections)) {
            this.cleanupConnection(room);
        }
    }

    stop() {
        this.cleanupAllConnections();
        if (this.ws) {
            this.ws.close();
        }
    }
}

// Make available globally
window.WebRTCViewer = WebRTCViewer;
