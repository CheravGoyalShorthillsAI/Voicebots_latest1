import os
import json
import time
import hashlib
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Callable, Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from functools import wraps

import boto3
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv(override=True)

# Configure logging
logger = logging.getLogger(__name__)
api_key="hvhvvchvcbhaaynuydbudvwydvwydvwdyvwd"

@dataclass
class TokenUsage:
    """Track token usage statistics"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_estimate: float = 0.0

@dataclass
class CacheEntry:
    """Cache entry for storing LLM responses"""
    response: str
    token_usage: TokenUsage
    timestamp: datetime
    ttl_minutes: int = 60
    
    def is_expired(self) -> bool:
        return datetime.now() > self.timestamp + timedelta(minutes=self.ttl_minutes)

@dataclass
class ConversationMetrics:
    """Track conversation performance metrics"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_tokens_used: int = 0
    average_response_time: float = 0.0
    errors_count: int = 0
    
    @property
    def cache_hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.cache_hits / self.total_requests) * 100

class ResponseCache:
    """In-memory cache for LLM responses with TTL support"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 60):
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.access_times: Dict[str, datetime] = {}
    
    def _generate_key(self, text: str, context: List[Dict[str, Any]]) -> str:
        """Generate cache key from input text and context"""
        context_str = json.dumps(context, sort_keys=True) if context else ""
        combined = f"{text}:{context_str}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    def get(self, text: str, context: List[Dict[str, Any]] = None) -> Optional[CacheEntry]:
        """Get cached response if available and not expired"""
        key = self._generate_key(text, context or [])
        
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        if entry.is_expired():
            del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]
            return None
        
        self.access_times[key] = datetime.now()
        return entry
    
    def set(self, text: str, response: str, token_usage: TokenUsage, 
            context: List[Dict[str, Any]] = None, ttl_minutes: Optional[int] = None):
        """Cache response with optional TTL"""
        key = self._generate_key(text, context or [])
        
        # Evict oldest entries if cache is full
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        entry = CacheEntry(
            response=response,
            token_usage=token_usage,
            timestamp=datetime.now(),
            ttl_minutes=ttl_minutes or self.default_ttl
        )
        
        self.cache[key] = entry
        self.access_times[key] = datetime.now()
    
    def _evict_oldest(self):
        """Evict oldest accessed entries"""
        if not self.access_times:
            return
            
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[oldest_key]
        del self.access_times[oldest_key]
    
    def clear_expired(self):
        """Remove all expired entries"""
        expired_keys = [
            key for key, entry in self.cache.items() 
            if entry.is_expired()
        ]
        
        for key in expired_keys:
            del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "total_entries": len(self.cache),
            "max_size": self.max_size,
            "expired_entries": sum(1 for entry in self.cache.values() if entry.is_expired())
        }

class EnhancedLLMProcessor:
    """Enhanced LLM processor with caching, error handling, and metrics"""
    
    def __init__(self):
        self.llm = self._initialize_llm()
        self.polly_client = self._initialize_polly()
        self.cache = ResponseCache(max_size=500, default_ttl=30)  # 30 min cache
        self.metrics = ConversationMetrics()
        self._setup_periodic_cleanup()
    
    def _initialize_llm(self) -> AzureChatOpenAI:
        """Initialize Azure OpenAI client with enhanced configuration"""
        try:
            return AzureChatOpenAI(
                deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                streaming=True,
                temperature=0.7,
                max_tokens=500,
                timeout=30,  # 30 second timeout
                max_retries=3
            )
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI: {e}")
            raise
    
    def _initialize_polly(self) -> boto3.client:
        """Initialize Amazon Polly client with enhanced configuration"""
        try:
            return boto3.client(
                "polly",
                region_name=os.getenv("AWS_REGION", "us-east-1"),
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
            )
        except Exception as e:
            logger.error(f"Failed to initialize Amazon Polly: {e}")
            raise
    
    def _setup_periodic_cleanup(self):
        """Setup periodic cache cleanup"""
        async def cleanup_task():
            while True:
                await asyncio.sleep(300)  # Clean every 5 minutes
                try:
                    self.cache.clear_expired()
                    logger.debug("Cache cleanup completed")
                except Exception as e:
                    logger.error(f"Cache cleanup error: {e}")
        
        asyncio.create_task(cleanup_task())
    
    
    def _extract_token_usage(self, response_metadata: Dict[str, Any]) -> TokenUsage:
        """Extract token usage from LLM response metadata"""
        usage_info = response_metadata.get('token_usage', {})
        
        token_usage = TokenUsage(
            prompt_tokens=usage_info.get('prompt_tokens', 0),
            completion_tokens=usage_info.get('completion_tokens', 0),
            total_tokens=usage_info.get('total_tokens', 0)
        )
        
        token_usage.cost_estimate = self._calculate_token_cost(token_usage)
        return token_usage
    
    async def get_audio_with_retry(self, text: str, max_retries: int = 3) -> bytes:
        """Generate audio with retry logic and enhanced error handling"""
        if not text.strip():
            logger.warning("Empty text provided for audio generation")
            return b''
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                # Enhanced Polly parameters
                response = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: self.polly_client.synthesize_speech(
                        Text=text[:3000],  # Polly character limit
                        OutputFormat="pcm",
                        VoiceId=os.getenv("POLLY_VOICE_ID", "Joanna"),
                        Engine="neural",
                        SampleRate="16000",
                        TextType="text"
                    )
                )
                
                audio_data = response["AudioStream"].read()
                generation_time = time.time() - start_time
                
                logger.info(f"Audio generated: {len(text)} chars -> {len(audio_data)} bytes in {generation_time:.2f}s")
                return audio_data
                
            except Exception as e:
                logger.warning(f"Audio generation attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Audio generation failed after {max_retries} attempts")
                    return b''
                await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
        
        return b''
    
    def _prepare_conversation_context(self, file_path: str, max_history: int = 3) -> List[Dict[str, Any]]:
        """Load and prepare conversation context with enhanced error handling"""
        try:
            if not os.path.exists(file_path):
                return []
            
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            if not isinstance(data, list):
                logger.warning(f"Invalid conversation file format: {file_path}")
                return []
            
            # Return last N conversations
            recent_conversations = data[-max_history:] if data else []
            return [conv for conv in recent_conversations if isinstance(conv, dict)]
            
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading conversation context from {file_path}: {e}")
            return []
    
    def _build_enhanced_prompt(self, user_text: str, context: List[Dict[str, Any]]) -> str:
        """Build enhanced prompt with better context handling"""
        base_instructions = """You are an advanced AI voice assistant. Provide helpful, accurate, and contextually appropriate responses. Keep responses conversational and concise (under 100 words unless more detail is specifically requested).

Guidelines:
- Handle transcription errors gracefully - focus on the intended meaning
- Provide natural, conversational responses
- Ask for clarification when the request is ambiguous
- Be helpful and informative while staying concise"""
        
        if not context:
            return f"{base_instructions}\n\nUser: {user_text}\nAssistant:"
        
        # Build conversation history
        history_text = []
        for conv in context:
            if isinstance(conv, dict):
                human_msg = conv.get('Human_Message', '').strip()
                ai_msg = conv.get('AI_Message', '').strip()
                if human_msg and ai_msg:
                    history_text.append(f"User: {human_msg}")
                    history_text.append(f"Assistant: {ai_msg}")
        
        history_section = "\n".join(history_text[-6:])  # Last 3 exchanges
        
        return f"""{base_instructions}

Previous conversation:
{history_section}

Current message:
User: {user_text}
Assistant:"""
    
    async def process_with_enhanced_features(
        self,
        stt_text: str,
        file_path: str,
        streaming_callback: Callable[[str, int, bytes], None],
        use_cache: bool = True,
        max_response_tokens: int = 150
    ) -> str:
        """
        Process text with LLM using enhanced features including caching and metrics
        """
        try:
            request_start = time.time()
            self.metrics.total_requests += 1
            
            # Load conversation context
            context = self._prepare_conversation_context(file_path)
            
            # Check cache first
            if use_cache:
                cached_entry = self.cache.get(stt_text, context)
                if cached_entry:
                    logger.info(f"Cache hit for request: {stt_text[:50]}...")
                    self.metrics.cache_hits += 1
                    
                    # Stream cached response
                    await self._stream_cached_response(cached_entry, streaming_callback)
                    return cached_entry.response
                
                self.metrics.cache_misses += 1
            
            # Build enhanced prompt
            prompt = self._build_enhanced_prompt(stt_text, context)
            
            # Process with LLM
            full_response = ""
            token_count = 0
            chunk_buffer = ""
            sequence = 0
            first_token_time = None
            
            try:
                # Stream LLM response with enhanced error handling
                async for chunk in self.llm.astream(input=prompt):
                    if first_token_time is None:
                        first_token_time = time.time()
                        logger.info(f"First token received in {first_token_time - request_start:.2f}s")
                    
                    content = chunk.content
                    if not content:
                        continue
                    
                    full_response += content
                    chunk_buffer += content
                    token_count += 1
                    
                    # Send chunks every 8-12 tokens for natural speech
                    if token_count % 10 == 0 or len(chunk_buffer) > 100:
                        sequence += 1
                        audio_data = await self.get_audio_with_retry(chunk_buffer)
                        await streaming_callback(chunk_buffer, sequence, audio_data)
                        
                        logger.debug(f"Streamed chunk {sequence}: '{chunk_buffer.strip()}'")
                        chunk_buffer = ""
                    
                    # Respect token limits
                    if len(full_response.split()) > max_response_tokens:
                        logger.info("Response token limit reached")
                        break
                        
            except asyncio.CancelledError:
                logger.info("LLM streaming was cancelled")
                raise
            except Exception as e:
                logger.error(f"Error during LLM streaming: {e}")
                self.metrics.errors_count += 1
                raise
            
            # Send remaining tokens
            if chunk_buffer.strip():
                sequence += 1
                audio_data = await self.get_audio_with_retry(chunk_buffer)
                await streaming_callback(chunk_buffer, sequence, audio_data)
            
            # Calculate metrics
            processing_time = time.time() - request_start
            token_usage = TokenUsage(
                prompt_tokens=len(prompt.split()) * 1.3,  # Rough estimate
                completion_tokens=len(full_response.split()),
                total_tokens=len(prompt.split()) * 1.3 + len(full_response.split())
            )
            token_usage.cost_estimate = self._calculate_token_cost(token_usage)
            
            # Update metrics
            self.metrics.total_tokens_used += int(token_usage.total_tokens)
            self.metrics.average_response_time = (
                (self.metrics.average_response_time * (self.metrics.total_requests - 1) + processing_time) /
                self.metrics.total_requests
            )
            
            # Cache the response
            if use_cache and full_response:
                self.cache.set(stt_text, full_response, token_usage, context, ttl_minutes=30)
            
            # Save conversation
            await self._save_conversation_async(file_path, stt_text, full_response)
            
            logger.info(f"LLM processing completed in {processing_time:.2f}s")
            logger.info(f"Token usage: {token_usage.total_tokens:.0f}, Cost: ${token_usage.cost_estimate:.4f}")
            
            return full_response
            
        except asyncio.CancelledError:
            logger.info("LLM processing cancelled by user")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in LLM processing: {e}")
            self.metrics.errors_count += 1
            
            # Return error response
            error_response = f"I apologize, but I encountered an error processing your request. Please try again."
            error_audio = await self.get_audio_with_retry(error_response)
            await streaming_callback(error_response, 1, error_audio)
            return error_response
    
    async def _stream_cached_response(
        self, 
        cached_entry: CacheEntry, 
        streaming_callback: Callable[[str, int, bytes], None]
    ):
        """Stream cached response to maintain consistent behavior"""
        response = cached_entry.response
        words = response.split()
        
        chunk_size = 8  # Words per chunk
        sequence = 0
        
        for i in range(0, len(words), chunk_size):
            sequence += 1
            chunk = " ".join(words[i:i + chunk_size])
            
            # Generate audio for chunk
            audio_data = await self.get_audio_with_retry(chunk)
            await streaming_callback(chunk, sequence, audio_data)
            
            # Small delay to simulate streaming
            await asyncio.sleep(0.1)
    
    async def _save_conversation_async(self, file_path: str, human_message: str, ai_message: str):
        """Save conversation asynchronously with enhanced error handling"""
        try:
            conversation_entry = {
                "timestamp": datetime.now().isoformat(),
                "Human_Message": human_message,
                "AI_Message": ai_message,
                "token_count": len(ai_message.split()),
                "processing_time": time.time()
            }
            
            # Use thread executor for file I/O
            await asyncio.get_running_loop().run_in_executor(
                None,
                self._save_conversation_sync,
                file_path,
                conversation_entry
            )
            
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
    
    def _save_conversation_sync(self, file_path: str, conversation_entry: Dict[str, Any]):
        """Synchronous conversation saving with file locking"""
        import fcntl  # For file locking on Unix systems
        
        try:
            # Load existing data
            data = []
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        fcntl.flock(file.fileno(), fcntl.LOCK_SH)  # Shared lock for reading
                        data = json.load(file)
                        if not isinstance(data, list):
                            data = []
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning(f"Error reading conversation file: {e}")
                    data = []
            
            # Add new entry
            data.append(conversation_entry)
            
            # Keep only last 50 conversations to prevent file bloat
            if len(data) > 50:
                data = data[-50:]
            
            # Save with exclusive lock
            with open(file_path, 'w', encoding='utf-8') as file:
                fcntl.flock(file.fileno(), fcntl.LOCK_EX)  # Exclusive lock for writing
                json.dump(data, file, indent=2, ensure_ascii=False)
            
            logger.debug(f"Conversation saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Error in synchronous conversation save: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive processor metrics"""
        cache_stats = self.cache.get_stats()
        
        return {
            "conversation_metrics": asdict(self.metrics),
            "cache_stats": cache_stats,
            "performance": {
                "cache_hit_rate": f"{self.metrics.cache_hit_rate:.1f}%",
                "average_response_time": f"{self.metrics.average_response_time:.2f}s",
                "total_cost_estimate": f"${(self.metrics.total_tokens_used * 0.002 / 1000):.4f}"
            }
        }
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.metrics = ConversationMetrics()
        logger.info("Metrics reset")

# Global processor instance
_processor_instance = None

def get_processor() -> EnhancedLLMProcessor:
    """Get singleton processor instance"""
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = EnhancedLLMProcessor()
    return _processor_instance

# Backward compatibility function
async def process_with_llm(
    stt_text: str,
    file_path: str,
    streaming_callback: Callable[[str, int, bytes], None]
) -> str:
    """
    Backward compatible function that uses enhanced processor
    """
    processor = get_processor()
    return await processor.process_with_enhanced_features(
        stt_text=stt_text,
        file_path=file_path,
        streaming_callback=streaming_callback,
        use_cache=True
    )

# Example usage and testing
if __name__ == "__main__":
    async def example_callback(token: str, sequence: int, audio_data: bytes):
        print(f"Sequence {sequence}: {token} (Audio: {len(audio_data)} bytes)")
    
    async def main():
        processor = get_processor()
        
        # Test with sample input
        result = await processor.process_with_enhanced_features(
            stt_text="What is the capital of France?",
            streaming_callback=example_callback,
            use_cache=True
        )
        
        print(f"\nFinal result: {result}")
        print(f"\nMetrics: {json.dumps(processor.get_metrics(), indent=2)}")
    
    if __name__ == "__main__":
        asyncio.run(main())
