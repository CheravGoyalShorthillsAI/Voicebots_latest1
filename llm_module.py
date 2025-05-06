import os
import json
import time
import asyncio
import boto3
from typing import Callable
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

# Initialize LLM client
llm = AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    streaming=True,
)

# Initialize Amazon Polly client
polly_client = boto3.client(
    "polly",
    region_name="us-east-1",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)

def append_to_json(filename: str, new_data: dict):
    """Append conversation data to a JSON file."""
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    if not isinstance(data, list):
        data = []

    data.append(new_data)
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"Data appended to {filename} successfully.")

async def get_audio(text: str) -> bytes:
    """Fetches speech audio from Amazon Polly asynchronously."""
    if not text.strip():
        print("[get_audio] Empty text, skipping")
        return b''

    print(f"[get_audio] Fetching audio for: {text}")
    loop = asyncio.get_running_loop()
    try:
        start_time = time.time()
        response = await loop.run_in_executor(None, lambda: polly_client.synthesize_speech(
            Text=text,
            OutputFormat="pcm",
            VoiceId="Joanna",
            Engine="neural",
            SampleRate="16000"
        ))
        audio_data = response["AudioStream"].read()
        print(f"[TTS] Generation time: {time.time() - start_time:.2f}s")
        return audio_data
    except Exception as e:
        print(f"[get_audio] Error fetching audio: {e}")
        return b''

async def process_with_llm(stt_text: str, file_path: str, streaming_callback: Callable[[str, int, bytes], None]) -> str:
    """
    Process transcribed text with Azure OpenAI in streaming mode and convert to audio.

    Args:
        stt_text: Transcribed speech input.
        file_path: Path to JSON file for conversation history.
        streaming_callback: Function to handle streamed tokens and audio (takes token, sequence number, audio data).

    Returns:
        Empty string (output is handled via callback).
    """
    try:
        start_time = time.time()
        first_token_time = None

        # Prepare prompt based on conversation history
        if not os.path.exists(file_path):
            prompt = f"""I will ask you some questions. Give me the most appropriate response you can find: {stt_text}

            Instructions: Question is generated using real-time transcription of an audio stream, so make sure
            you try to get the basic meaning of the question and ignore repeated words.
            """
        else:
            with open(file_path, 'r') as file:
                data = json.load(file)

            if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                conversation_history = data[-3:]  # Use last 3 conversations
            else:
                conversation_history = []

            history_text = "\n".join(
                [f"Human: {conv['Human_Message']}\nAI: {conv['AI_Message']}" for conv in conversation_history]
            )
            prompt = f"""
            I will ask you some questions. Give me the most appropriate response you can find: {stt_text}.
            Please give only relevant responses and take into account our previous history.

            Instructions:
            Question is generated using real-time transcription of an audio stream, so make sure
            you try to get the basic meaning of the question and ignore repeated words.

            Here are previous conversations:
            {history_text}
            """

        # Stream LLM response
        full_response = ""
        count = 0
        local_tokens = ""
        for chunk in llm.stream(input=prompt):
            if first_token_time is None:
                first_token_time = time.time()
                print(f"[LLM] First token time: {first_token_time - start_time:.2f}s")
            token = chunk.content
            full_response += token
            count += 1
            local_tokens += token
            if count % 10 == 0:
                print(f"Local tokens: {local_tokens}")
                audio_data = await get_audio(local_tokens)
                await streaming_callback(local_tokens, count // 10, audio_data)
                local_tokens = ""

        # Send remaining tokens
        if local_tokens:
            audio_data = await get_audio(local_tokens)
            await streaming_callback(local_tokens, (count // 10) + 1, audio_data)

        print(f"[LLM] Processing time: {time.time() - start_time:.2f}s")

        # Save conversation
        conversation = {
            "Human_Message": stt_text,
            "AI_Message": full_response
        }
        append_to_json(file_path, conversation)

        return ""

    except Exception as e:
        print(f"LLM Error: {str(e)}")
        return f"LLM Error: {str(e)}"

# Example usage
if __name__ == "__main__":
    async def example_callback(token: str, sequence: int, audio_data: bytes):
        """Example callback to print tokens and audio data."""
        print(f"Sequence {sequence}: {token} (Audio: {len(audio_data)} bytes)", end="", flush=True)

    import asyncio
    asyncio.run(process_with_llm(
        stt_text="Tell me about the capital of India in 200 words?",
        file_path="demo.json",
        streaming_callback=example_callback
    ))