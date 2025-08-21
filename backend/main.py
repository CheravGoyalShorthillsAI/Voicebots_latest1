import asyncio
import os
import json
import time
import logging
from datetime import datetime
from io import BytesIO
from typing import Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketState
from deepgram import DeepgramClient, DeepgramClientOptions, LiveTranscriptionEvents, LiveOptions

from config import config
from llm_module import process_with_llm

# Setup logging
config.setup_logging()
logger = logging.getLogger(__name__)

# Initialize FastAPI app with metadata
app = FastAPI(
    title="Voicebot Latest API",
    description="Advanced Real-time Speech Recognition and AI Response System",
    version="1.2.0",
    docs_url="/docs" if config.DEBUG else None,
    redoc_url="/redoc" if config.DEBUG else None
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Application metrics
app.state.metrics = {
    "total_connections": 0,
    "active_connections": 0,
    "total_messages_processed": 0,
    "errors_count": 0,
    "uptime_start": datetime.now(),
    "deepgram_calls": 0,
    "llm_calls": 0
}

class ConnectionManager:
    """Manage WebSocket connections with better tracking"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_data: Dict[str, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.connection_data[client_id] = {
            "connected_at": datetime.now(),
            "messages_count": 0,
            "last_activity": datetime.now()
        }
        app.state.metrics["total_connections"] += 1
        app.state.metrics["active_connections"] += 1
        logger.info(f"Client {client_id} connected. Active connections: {len(self.active_connections)}")
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            del self.connection_data[client_id]
            app.state.metrics["active_connections"] -= 1
            logger.info(f"Client {client_id} disconnected. Active connections: {len(self.active_connections)}")
    
    def update_activity(self, client_id: str):
        if client_id in self.connection_data:
            self.connection_data[client_id]["last_activity"] = datetime.now()
            self.connection_data[client_id]["messages_count"] += 1

manager = ConnectionManager()

@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    logger.info("🚀 Voicebot application starting up...")
    
    # Validate configuration
    missing_config = config.validate_config()
    if missing_config:
        logger.error(f"❌ Missing required configuration: {', '.join(missing_config)}")
        raise HTTPException(status_code=500, detail="Missing required configuration")
    
    logger.info("✅ Configuration validated successfully")
    logger.info(f"🎵 Audio settings: {config.SAMPLE_RATE}Hz, {config.CHANNELS} channel(s)")
    logger.info(f"🤖 LLM model: {config.LLM_MODEL}")
    logger.info("✅ Application startup completed")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    logger.info("🛑 Voicebot application shutting down...")
    # Cleanup active connections
    for client_id in list(manager.active_connections.keys()):
        manager.disconnect(client_id)
    logger.info("✅ Application shutdown completed")

# Health Check Endpoints
@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.2.0"
    }

@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with system information"""
    uptime = datetime.now() - app.state.metrics["uptime_start"]
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.2.0",
        "uptime_seconds": uptime.total_seconds(),
        "metrics": app.state.metrics,
        "active_connections": len(manager.active_connections),
        "configuration": {
            "sample_rate": config.SAMPLE_RATE,
            "channels": config.CHANNELS,
            "model": config.LLM_MODEL,
            "debug_mode": config.DEBUG
        }
    }

@app.get("/metrics")
async def get_metrics():
    """Get application metrics"""
    if not config.METRICS_ENABLED:
        raise HTTPException(status_code=404, detail="Metrics disabled")
    
    uptime = datetime.now() - app.state.metrics["uptime_start"]
    return {
        "metrics": app.state.metrics,
        "uptime_seconds": uptime.total_seconds(),
        "connections": {
            "active": len(manager.active_connections),
            "total": app.state.metrics["total_connections"]
        }
    }

@app.get("/")
async def get_index():
    """Serve the main frontend page"""
    try:
        return FileResponse("frontend/index.html")
    except FileNotFoundError:
        logger.error("Frontend index.html not found")
        raise HTTPException(status_code=404, detail="Frontend not found")

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str = "default"):
    """Enhanced WebSocket endpoint with better connection management"""
    
    try:
        await manager.connect(websocket, client_id)
        logger.info(f"🔵 Client {client_id} connected to WebSocket")

        # Initialize Deepgram with enhanced error handling
        try:
            deepgram = DeepgramClient(
                config.DEEPGRAM_API_KEY, 
                DeepgramClientOptions(options={"keepalive": "true"})
            )
            dg_connection = deepgram.listen.asyncwebsocket.v("1")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Deepgram: {e}")
            await websocket.close(code=1011, reason="Service initialization failed")
            return

        # Enhanced processing with cleanup
        connection_closed_once = False
        final_transcripts = []
        audio_queue = asyncio.Queue()
        llm_task = None
        audio_task = None

        async def cleanup_tasks():
            """Enhanced task cleanup"""
            nonlocal llm_task, audio_task
            
            if llm_task and not llm_task.done():
                llm_task.cancel()
                try:
                    await llm_task
                except asyncio.CancelledError:
                    logger.info(f"🟢 LLM task cancelled for {client_id}")
            
            if audio_task and not audio_task.done():
                audio_task.cancel()
                try:
                    await audio_task
                except asyncio.CancelledError:
                    logger.info(f"🟢 Audio task cancelled for {client_id}")
            
            # Clear audio queue
            while not audio_queue.empty():
                try:
                    audio_queue.get_nowait()
                    audio_queue.task_done()
                except asyncio.QueueEmpty:
                    break

        # Enhanced event handlers
        async def on_open(_, msg, **kwargs):
            logger.info(f"🔵 Deepgram opened for {client_id}: {msg}")
            app.state.metrics["deepgram_calls"] += 1

        async def on_close(_, **kwargs):
            nonlocal connection_closed_once
            if not connection_closed_once:
                logger.info(f"🔴 Deepgram closed for {client_id}")
                connection_closed_once = True

        async def on_transcript(_, result, **kwargs):
            nonlocal llm_task, audio_task
            
            try:
                transcript = result.channel.alternatives[0].transcript
                if not transcript:
                    return

                manager.update_activity(client_id)
                app.state.metrics["total_messages_processed"] += 1
                
                # Enhanced interruption handling
                if audio_task and not audio_task.done() and result.is_final:
                    if websocket.client_state == WebSocketState.CONNECTED:
                        try:
                            await websocket.send_text(json.dumps({
                                "type": "pause_playback",
                                "text": "🎙️ Processing...",
                                "client_id": client_id,
                                "timestamp": datetime.now().isoformat()
                            }))
                            await cleanup_tasks()
                            audio_task = asyncio.create_task(stream_audio())
                        except Exception as e:
                            logger.error(f"❗ Error in interruption handling: {e}")

                if result.is_final:
                    final_transcripts.append(transcript)
                    
                    if websocket.client_state == WebSocketState.CONNECTED:
                        try:
                            await websocket.send_text(json.dumps({
                                "type": "transcript",
                                "text": transcript,
                                "is_final": True,
                                "client_id": client_id,
                                "timestamp": datetime.now().isoformat()
                            }))
                        except Exception as e:
                            logger.error(f"❗ Error sending transcript: {e}")

                    if result.speech_final:
                        full_transcript = " ".join(final_transcripts)
                        logger.info(f"🟢 Processing complete: {full_transcript}")
                        
                        await cleanup_tasks()
                        
                        # Process with enhanced error handling
                        async def llm_callback(token: str, sequence: int, audio_data: bytes):
                            if token and websocket.client_state == WebSocketState.CONNECTED:
                                try:
                                    await websocket.send_text(json.dumps({
                                        "type": "llm_response",
                                        "text": token,
                                        "sequence": sequence,
                                        "client_id": client_id,
                                        "timestamp": datetime.now().isoformat()
                                    }))
                                except Exception as e:
                                    logger.error(f"❗ Error sending LLM response: {e}")

                            if audio_data:
                                await audio_queue.put(audio_data)

                        try:
                            audio_task = asyncio.create_task(stream_audio())
                            llm_task = asyncio.create_task(
                                process_with_llm(
                                    stt_text=full_transcript,
                                    file_path=f"{config.CONVERSATIONS_FILE}_{client_id}",
                                    streaming_callback=llm_callback
                                )
                            )
                            app.state.metrics["llm_calls"] += 1
                            await llm_task
                            final_transcripts.clear()
                        except asyncio.CancelledError:
                            logger.info(f"🟢 Processing cancelled for {client_id}")
                        except Exception as e:
                            logger.error(f"❗ LLM processing error: {e}")
                            app.state.metrics["errors_count"] += 1
                        
            except Exception as e:
                logger.error(f"❗ Transcript handler error: {e}")
                app.state.metrics["errors_count"] += 1

        async def stream_audio():
            """Enhanced audio streaming"""
            while True:
                try:
                    audio_data = await audio_queue.get()
                    if audio_data is None:
                        break
                        
                    if websocket.client_state == WebSocketState.CONNECTED:
                        buffer = BytesIO(audio_data)
                        chunk_size = int(0.05 * config.SAMPLE_RATE * 2)
                        
                        while True:
                            chunk = buffer.read(chunk_size)
                            if not chunk:
                                break
                            await websocket.send_bytes(chunk)
                            await asyncio.sleep(0.02)
                    
                    audio_queue.task_done()
                except Exception as e:
                    logger.error(f"❗ Audio streaming error: {e}")

        # Register handlers
        dg_connection.on(LiveTranscriptionEvents.Open, on_open)
        dg_connection.on(LiveTranscriptionEvents.Close, on_close)
        dg_connection.on(LiveTranscriptionEvents.Transcript, on_transcript)

        # Enhanced options
        options = LiveOptions(
            model=config.LLM_MODEL,
            language=config.LANGUAGE,
            smart_format=True,
            encoding=config.AUDIO_FORMAT,
            channels=config.CHANNELS,
            sample_rate=config.SAMPLE_RATE,
            interim_results=True,
            utterance_end_ms="1000",
            vad_events=True,
            punctuate=False
        )

        await dg_connection.start(options)
        audio_task = asyncio.create_task(stream_audio())

        # Main message loop
        while True:
            try:
                data = await websocket.receive()
                manager.update_activity(client_id)
                
                if 'text' in data:
                    try:
                        message = json.loads(data['text'])
                        if message.get('type') == 'interrupt':
                            logger.info(f"🟢 Interrupt received from {client_id}")
                            await cleanup_tasks()
                            final_transcripts.clear()
                            audio_task = asyncio.create_task(stream_audio())
                    except json.JSONDecodeError:
                        logger.warning(f"⚠️ Invalid JSON from {client_id}")
                        
                elif 'bytes' in data:
                    await dg_connection.send(data['bytes'])

            except WebSocketDisconnect:
                logger.info(f"⚡ Client {client_id} disconnected")
                break
            except Exception as e:
                logger.error(f"❌ WebSocket error for {client_id}: {e}")
                break

    except Exception as e:
        logger.error(f"❌ Connection error for {client_id}: {e}")
        app.state.metrics["errors_count"] += 1
    
    finally:
        # Comprehensive cleanup
        logger.info(f"🧹 Cleaning up {client_id}")
        manager.disconnect(client_id)
        
        try:
            await dg_connection.finish()
        except:
            pass

        try:
            await audio_queue.put(None)
            if audio_task and not audio_task.done():
                audio_task.cancel()
                await audio_task
        except:
            pass

        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close()

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting Voicebot server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=config.DEBUG)