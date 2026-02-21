// Face Analysis Debugger - Application Logic

let mqttClient = null;
let isConnected = false;
let lastFrameTime = 0;
let frameCount = 0;
let fpsUpdateInterval = null;
let currentFaces = [];

// DOM Elements
const mqttStatus = document.getElementById('mqttStatus');
const videoStatus = document.getElementById('videoStatus');
const videoElement = document.getElementById('videoElement');
const overlayCanvas = document.getElementById('overlayCanvas');
const videoPlaceholder = document.getElementById('videoPlaceholder');
const connectBtn = document.getElementById('connectBtn');
const disconnectBtn = document.getElementById('disconnectBtn');
const faceCountEl = document.getElementById('faceCount');
const fpsEl = document.getElementById('fps');
const inferenceTimeEl = document.getElementById('inferenceTime');
const frameIdEl = document.getElementById('frameId');
const faceListEl = document.getElementById('faceList');
const logContainer = document.getElementById('logContainer');

// Canvas context for drawing overlays
const ctx = overlayCanvas.getContext('2d');

// Emotion colors
const emotionColors = {
    'neutral': '#808080',
    'happiness': '#FFD700',
    'surprise': '#FF69B4',
    'sadness': '#4169E1',
    'anger': '#FF4500',
    'disgust': '#9ACD32',
    'fear': '#800080',
    'contempt': '#8B4513'
};

// Logging
function log(message, type = 'info') {
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    logContainer.appendChild(entry);
    logContainer.scrollTop = logContainer.scrollHeight;

    // Keep only last 100 log entries
    while (logContainer.children.length > 100) {
        logContainer.removeChild(logContainer.firstChild);
    }
}

// Connect to MQTT and video stream
function connect() {
    const mqttUrl = document.getElementById('mqttUrl').value;
    const mqttTopic = document.getElementById('mqttTopic').value;
    const rtspUrl = document.getElementById('rtspUrl').value;

    // Connect to MQTT
    connectMQTT(mqttUrl, mqttTopic);

    // Connect to video stream
    connectVideo(rtspUrl);

    connectBtn.disabled = true;
    disconnectBtn.disabled = false;
}

// Disconnect from all services
function disconnect() {
    if (mqttClient) {
        mqttClient.end();
        mqttClient = null;
    }

    videoElement.src = '';
    videoPlaceholder.style.display = 'block';
    videoStatus.classList.remove('connected');

    mqttStatus.classList.remove('connected');
    isConnected = false;

    if (fpsUpdateInterval) {
        clearInterval(fpsUpdateInterval);
        fpsUpdateInterval = null;
    }

    connectBtn.disabled = false;
    disconnectBtn.disabled = true;

    log('Disconnected', 'info');
}

// MQTT Connection
function connectMQTT(url, topic) {
    log(`Connecting to MQTT: ${url}`, 'info');

    try {
        mqttClient = mqtt.connect(url, {
            clientId: 'face-analysis-debugger-' + Math.random().toString(16).substr(2, 8),
            clean: true,
            connectTimeout: 10000,
            reconnectPeriod: 5000
        });

        mqttClient.on('connect', () => {
            log('MQTT connected', 'success');
            mqttStatus.classList.add('connected');
            isConnected = true;

            mqttClient.subscribe(topic, (err) => {
                if (err) {
                    log(`Failed to subscribe: ${err.message}`, 'error');
                } else {
                    log(`Subscribed to: ${topic}`, 'success');
                }
            });

            // Start FPS counter
            startFPSCounter();
        });

        mqttClient.on('message', (topic, message) => {
            try {
                const data = JSON.parse(message.toString());
                handleFaceData(data);
            } catch (e) {
                log(`Failed to parse message: ${e.message}`, 'error');
            }
        });

        mqttClient.on('error', (err) => {
            log(`MQTT error: ${err.message}`, 'error');
            mqttStatus.classList.remove('connected');
        });

        mqttClient.on('close', () => {
            log('MQTT connection closed', 'info');
            mqttStatus.classList.remove('connected');
            isConnected = false;
        });

    } catch (e) {
        log(`MQTT connection failed: ${e.message}`, 'error');
    }
}

// Video Connection (using WebRTC/WHEP or fallback to HLS)
function connectVideo(url) {
    log(`Connecting to video: ${url}`, 'info');

    videoPlaceholder.style.display = 'none';

    // Try different video sources
    if (url.includes('whep') || url.endsWith('/whep')) {
        // WebRTC WHEP
        connectWebRTC(url);
    } else if (url.endsWith('.m3u8')) {
        // HLS
        videoElement.src = url;
        videoElement.play().then(() => {
            log('HLS video connected', 'success');
            videoStatus.classList.add('connected');
        }).catch(e => {
            log(`Video play failed: ${e.message}`, 'error');
        });
    } else {
        // Try as direct URL or MediaMTX proxy
        fetch(url)
            .then(response => {
                if (response.ok) {
                    // Assume it's a video proxy endpoint
                    videoElement.src = url;
                    videoElement.play().then(() => {
                        log('Video stream connected', 'success');
                        videoStatus.classList.add('connected');
                    });
                }
            })
            .catch(e => {
                log(`Video connection failed: ${e.message}`, 'error');
                videoPlaceholder.style.display = 'block';
                videoPlaceholder.innerHTML = `
                    <p style="color: #f72585;">Video connection failed</p>
                    <p style="font-size: 0.8rem; margin-top: 10px;">
                        Try using a WebRTC proxy like MediaMTX<br>
                        or configure RTSP-to-WebSocket bridge
                    </p>
                `;
            });
    }

    // Resize canvas when video loads
    videoElement.onloadedmetadata = () => {
        resizeCanvas();
    };
}

// WebRTC connection (WHEP protocol)
async function connectWebRTC(whepUrl) {
    try {
        const pc = new RTCPeerConnection({
            iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
        });

        pc.addTransceiver('video', { direction: 'recvonly' });
        pc.addTransceiver('audio', { direction: 'recvonly' });

        pc.ontrack = (event) => {
            if (event.track.kind === 'video') {
                videoElement.srcObject = event.streams[0];
                videoStatus.classList.add('connected');
                log('WebRTC video connected', 'success');
            }
        };

        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);

        const response = await fetch(whepUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/sdp' },
            body: offer.sdp
        });

        if (response.ok) {
            const answer = await response.text();
            await pc.setRemoteDescription({ type: 'answer', sdp: answer });
        } else {
            throw new Error(`WHEP failed: ${response.status}`);
        }
    } catch (e) {
        log(`WebRTC connection failed: ${e.message}`, 'error');
    }
}

// Handle incoming face data
function handleFaceData(data) {
    currentFaces = data.faces || [];

    // Update statistics
    faceCountEl.textContent = data.face_count || 0;
    inferenceTimeEl.textContent = Math.round(data.inference_time_ms || 0);
    frameIdEl.textContent = data.frame_id || 0;

    // Update FPS counter
    frameCount++;

    // Update face list
    updateFaceList(currentFaces);

    // Draw overlays on canvas
    drawOverlays(currentFaces);
}

// Update face list panel
function updateFaceList(faces) {
    if (faces.length === 0) {
        faceListEl.innerHTML = '<div style="color: #666; text-align: center; padding: 20px;">No faces detected</div>';
        return;
    }

    faceListEl.innerHTML = faces.map(face => {
        const attrs = face;
        const emotionColor = emotionColors[attrs.emotion] || '#888';

        return `
            <div class="face-card">
                <div class="face-header">
                    <span class="face-id">Face #${face.id}</span>
                    <span class="face-confidence">${(face.confidence * 100).toFixed(1)}%</span>
                </div>
                <div class="face-attrs">
                    <div class="attr-item">
                        <span class="attr-label">Age</span>
                        <span class="attr-value">${attrs.age}</span>
                    </div>
                    <div class="attr-item">
                        <span class="attr-label">Gender</span>
                        <span class="attr-value">${attrs.gender} (${(attrs.gender_confidence * 100).toFixed(0)}%)</span>
                    </div>
                    <div class="attr-item" style="grid-column: span 2;">
                        <span class="attr-label">Emotion</span>
                        <span class="attr-value" style="color: ${emotionColor}">
                            ${attrs.emotion} (${(attrs.emotion_confidence * 100).toFixed(0)}%)
                        </span>
                    </div>
                </div>
                <div class="emotion-bar">
                    <div class="emotion-fill" style="width: ${attrs.emotion_confidence * 100}%; background: ${emotionColor}"></div>
                </div>
            </div>
        `;
    }).join('');
}

// Draw face boxes and labels on canvas
function drawOverlays(faces) {
    resizeCanvas();
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    const videoRect = videoElement.getBoundingClientRect();
    const canvasRect = overlayCanvas.getBoundingClientRect();

    // Calculate scale factors
    const scaleX = overlayCanvas.width;
    const scaleY = overlayCanvas.height;

    faces.forEach(face => {
        const bbox = face.bbox;

        // Convert normalized coordinates to canvas coordinates
        const x = bbox.x * scaleX;
        const y = bbox.y * scaleY;
        const w = bbox.w * scaleX;
        const h = bbox.h * scaleY;

        const emotionColor = emotionColors[face.emotion] || '#4cc9f0';

        // Draw bounding box
        ctx.strokeStyle = emotionColor;
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, w, h);

        // Draw label background
        const label = `${face.gender}, ${face.age} - ${face.emotion}`;
        ctx.font = '14px Arial';
        const textWidth = ctx.measureText(label).width;
        const labelHeight = 20;

        ctx.fillStyle = emotionColor;
        ctx.fillRect(x, y - labelHeight - 2, textWidth + 10, labelHeight);

        // Draw label text
        ctx.fillStyle = '#000';
        ctx.fillText(label, x + 5, y - 6);

        // Draw confidence bar
        const barWidth = w * face.confidence;
        ctx.fillStyle = emotionColor;
        ctx.fillRect(x, y + h + 2, barWidth, 3);
    });
}

// Resize canvas to match video
function resizeCanvas() {
    const rect = document.querySelector('.video-container').getBoundingClientRect();
    overlayCanvas.width = rect.width;
    overlayCanvas.height = rect.height;
}

// FPS Counter
function startFPSCounter() {
    frameCount = 0;
    lastFrameTime = Date.now();

    fpsUpdateInterval = setInterval(() => {
        const now = Date.now();
        const elapsed = (now - lastFrameTime) / 1000;
        const fps = frameCount / elapsed;

        fpsEl.textContent = fps.toFixed(1);

        frameCount = 0;
        lastFrameTime = now;
    }, 1000);
}

// Window resize handler
window.addEventListener('resize', () => {
    resizeCanvas();
    drawOverlays(currentFaces);
});

// Initialize
log('Face Analysis Debugger initialized', 'info');
log('Enter connection details and click Connect', 'info');
