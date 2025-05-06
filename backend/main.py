import asyncio
import os
import json
from io import BytesIO
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from starlette.websockets import WebSocketState
from deepgram import DeepgramClient, DeepgramClientOptions, LiveTranscriptionEvents, LiveOptions
from llm_module import process_with_llm

load_dotenv(override=True)
DEEPGRAM_API_KEY = os.getenv('DEEPGRAM_API_KEY')

app = FastAPI()

@app.get("/")
async def get_index():
    return FileResponse("frontend/index.html")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")

    # Set up Deepgram connection
    deepgram = DeepgramClient(DEEPGRAM_API_KEY, DeepgramClientOptions(options={"keepalive": "true"}))
    dg_connection = deepgram.listen.asyncwebsocket.v("1")

    connection_closed_once = False
    final_transcripts = []  # Accumulate transcripts until speech_final
    audio_queue = asyncio.Queue()  # Queue for audio chunks
    llm_task = None  # Track LLM processing task
    audio_task = None  # Track audio streaming task

    async def on_open(_, msg, **kwargs):
        print(f"🔵 Deepgram connection opened: {msg}")

    async def on_close(_, **kwargs):
        nonlocal connection_closed_once
        if not connection_closed_once:
            print("🔴 Deepgram connection closed")
            connection_closed_once = True

    async def on_transcript(_, result, **kwargs):
        nonlocal llm_task, audio_task
        transcript = result.channel.alternatives[0].transcript
        if not transcript:
            return

        if result.is_final:
            final_transcripts.append(transcript)
            # Send interim transcript to client
            if websocket.client_state == WebSocketState.CONNECTED:
                try:
                    await websocket.send_text(
                        f'{{"type": "transcript", "text": "{transcript}"}}'
                    )
                    print(f"🟢 Interim transcript sent: {transcript}")
                except Exception as send_err:
                    print(f"❗ Error sending interim transcript: {send_err}")

            if result.speech_final:
                full_transcript = " ".join(final_transcripts)
                print(f"🟢 Speech final transcript: {full_transcript}")

                # Send final transcript to client
                if websocket.client_state == WebSocketState.CONNECTED:
                    try:
                        await websocket.send_text(
                            f'{{"type": "transcript", "text": "{full_transcript}"}}'
                        )
                    except Exception as send_err:
                        print(f"❗ Error sending final transcript: {send_err}")

                # Cancel any ongoing tasks
                if llm_task and not llm_task.done():
                    llm_task.cancel()
                    try:
                        await llm_task
                    except asyncio.CancelledError:
                        print("🟢 Previous LLM task cancelled")
                if audio_task and not audio_task.done():
                    audio_task.cancel()
                    try:
                        await audio_task
                    except asyncio.CancelledError:
                        print("🟢 Previous audio task cancelled")

                # Clear audio queue
                while not audio_queue.empty():
                    try:
                        audio_queue.get_nowait()
                        audio_queue.task_done()
                    except asyncio.QueueEmpty:
                        break

                # Process with LLM and stream audio
                async def llm_callback(token: str, sequence: int, audio_data: bytes):
                    if token and websocket.client_state == WebSocketState.CONNECTED:
                        try:
                            await websocket.send_text(
                                f'{{"type": "llm_response", "text": "{token}"}}'
                            )
                            print(f"🟢 LLM response sent (sequence {sequence}): {token}")
                        except Exception as send_err:
                            print(f"❗ Error sending LLM response: {send_err}")

                    if audio_data:
                        await audio_queue.put(audio_data)

                try:
                    # Start new audio streaming task
                    audio_task = asyncio.create_task(stream_audio())
                    llm_task = asyncio.create_task(
                        process_with_llm(
                            stt_text=full_transcript,
                            file_path="conversations.json",
                            streaming_callback=llm_callback
                        )
                    )
                    await llm_task
                    final_transcripts.clear()  # Reset for next utterance
                except asyncio.CancelledError:
                    print("🟢 LLM task cancelled due to interruption")
                    final_transcripts.clear()
                except Exception as llm_err:
                    print(f"❗ LLM processing error: {llm_err}")

    async def on_metadata(_, metadata, **kwargs):
        print(f"📊 Metadata received: {metadata}")

    async def on_error(_, err, **kwargs):
        print(f"❗ Deepgram error: {err}")

    async def on_warning(_, warning, **kwargs):
        print(f"⚠️ Deepgram warning: {warning}")

    # Register handlers
    dg_connection.on(LiveTranscriptionEvents.Open, on_open)
    dg_connection.on(LiveTranscriptionEvents.Close, on_close)
    dg_connection.on(LiveTranscriptionEvents.Transcript, on_transcript)
    dg_connection.on(LiveTranscriptionEvents.Metadata, on_metadata)
    dg_connection.on(LiveTranscriptionEvents.Error, on_error)
    dg_connection.on(LiveTranscriptionEvents.Warning, on_warning)

    options = LiveOptions(
        model="nova-3",
        language="en-US",
        smart_format=True,
        encoding="linear16",
        channels=1,
        sample_rate=16000,
        interim_results=True,
        utterance_end_ms="1000",
        vad_events=True,
        punctuate=False
    )

    # Task to stream audio from queue
    async def stream_audio():
        while True:
            audio_data = await audio_queue.get()
            if audio_data is None:
                print("🟢 Audio queue terminated")
                break
            if websocket.client_state == WebSocketState.CONNECTED:
                try:
                    buffer = BytesIO(audio_data)
                    while True:
                        chunk = buffer.read(int(0.05 * 16000 * 2))  # 50ms chunks
                        if not chunk:
                            break
                        await websocket.send_bytes(chunk)
                        await asyncio.sleep(0.02)  # Control pacing
                except Exception as send_err:
                    print(f"❗ Error sending audio chunk: {send_err}")
            audio_queue.task_done()

    try:
        await dg_connection.start(options)
        # Start initial audio streaming task
        audio_task = asyncio.create_task(stream_audio())

        while True:
            data = await websocket.receive()
            if 'text' in data:
                message = json.loads(data['text'])
                if message.get('type') == 'interrupt':
                    print("🟢 Received interruption signal")
                    # Cancel ongoing tasks
                    if llm_task and not llm_task.done():
                        llm_task.cancel()
                        try:
                            await llm_task
                        except asyncio.CancelledError:
                            print("🟢 LLM task cancelled due to interruption")
                    if audio_task and not audio_task.done():
                        audio_task.cancel()
                        try:
                            await audio_task
                        except asyncio.CancelledError:
                            print("🟢 Audio task cancelled due to interruption")
                    # Clear audio queue
                    while not audio_queue.empty():
                        try:
                            audio_queue.get_nowait()
                            audio_queue.task_done()
                        except asyncio.QueueEmpty:
                            break
                    final_transcripts.clear()
                    # Start new audio streaming task
                    audio_task = asyncio.create_task(stream_audio())
                    continue
            elif 'bytes' in data:
                audio_chunk = data['bytes']
                await dg_connection.send(audio_chunk)

    except WebSocketDisconnect:
        print("⚡ Client disconnected")

    except Exception as e:
        print(f"❌ Unexpected error: {e}")

    finally:
        print("Cleaning up")
        try:
            await dg_connection.finish()
        except asyncio.CancelledError:
            pass
        except Exception as finish_err:
            print(f"⚠️ Error during Deepgram finish: {finish_err}")

        if not connection_closed_once:
            print("🔴 Deepgram connection closed")

        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close()

        # Signal end of audio and cancel streaming task
        await audio_queue.put(None)
        if audio_task and not audio_task.done():
            audio_task.cancel()
            try:
                await audio_task
            except asyncio.CancelledError:
                pass