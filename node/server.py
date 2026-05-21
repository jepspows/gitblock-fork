"""GitBlock Node Server — Serve AI models on the decentralized network."""
import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import NodeConfig
from .reputation import ReputationScore
from .rewards import RewardCalculator

logger = logging.getLogger("gitblock-node")


# ── Request / Response Models ──

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 1000
    stream: bool = False

class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: dict

class HealthResponse(BaseModel):
    status: str
    node_id: str
    uptime_hours: float
    reputation: float
    models: list[str]
    requests_served: int


# ── Application State ──

class NodeState:
    def __init__(self, config: NodeConfig):
        self.config = config
        self.node_id = str(uuid.uuid4())[:8]
        self.reputation = ReputationScore()
        self.rewards = RewardCalculator()
        self.start_time = time.time()
        self._lock = asyncio.Lock()
        self.active_requests = 0

    async def process_request(self, request: ChatRequest) -> ChatResponse:
        """Process an inference request."""
        async with self._lock:
            self.active_requests += 1

        start = time.time()
        try:
            # Simulate model inference (replace with real model loading)
            response_text = await self._run_inference(request)
            latency_ms = (time.time() - start) * 1000

            self.reputation.record_success(latency_ms)
            return ChatResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
                created=int(time.time()),
                model=request.model,
                choices=[ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason="stop",
                )],
                usage={
                    "prompt_tokens": sum(len(m.content.split()) for m in request.messages),
                    "completion_tokens": len(response_text.split()),
                    "total_tokens": 0,
                },
            )
        except Exception as e:
            self.reputation.record_failure()
            raise
        finally:
            async with self._lock:
                self.active_requests -= 1

    async def _run_inference(self, request: ChatRequest) -> str:
        """Run model inference. Override with actual model backend."""
        # Placeholder — in production, load and run the actual model
        await asyncio.sleep(0.1)  # Simulate processing
        return (
            f"[GitBlock Node {self.node_id}] "
            f"This is a placeholder response from model '{request.model}'. "
            f"Connect a real model backend (vLLM, llama.cpp, etc.) for production use."
        )


# ── FastAPI App ──

def create_app(config: Optional[NodeConfig] = None) -> FastAPI:
    """Create the FastAPI application."""
    config = config or NodeConfig.load()
    state = NodeState(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info(f"Node {state.node_id} starting on {config.listen_host}:{config.listen_port}")
        logger.info(f"Serving models: {', '.join(config.models)}")
        yield
        logger.info(f"Node {state.node_id} shutting down")

    app = FastAPI(
        title="GitBlock Node",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(
            status="ok",
            node_id=state.node_id,
            uptime_hours=state.reputation.uptime_hours,
            reputation=state.reputation.overall,
            models=config.models,
            requests_served=state.reputation.total_requests,
        )

    @app.post("/v1/chat/completions", response_model=ChatResponse)
    async def chat_completions(request: ChatRequest):
        if request.model not in config.models:
            raise HTTPException(404, f"Model '{request.model}' not available on this node")
        if state.active_requests >= config.max_concurrent:
            raise HTTPException(429, "Node at maximum capacity")
        return await state.process_request(request)

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [{"id": m, "object": "model", "owned_by": "gitblock"} for m in config.models],
        }

    @app.get("/status")
    async def status():
        return {
            "node_id": state.node_id,
            "reputation": state.reputation.to_dict(),
            "rewards_earned": state.rewards.calculate(
                state.reputation.total_requests,
                state.reputation.overall,
                state.reputation.uptime_hours,
            ),
            "active_requests": state.active_requests,
            "config": {
                "models": config.models,
                "max_concurrent": config.max_concurrent,
                "wallet": config.wallet_address,
            },
        }

    return app


def main():
    """Run the node server."""
    import uvicorn
    config = NodeConfig.load()
    logging.basicConfig(level=getattr(logging, config.log_level))
    app = create_app(config)
    uvicorn.run(app, host=config.listen_host, port=config.listen_port)


if __name__ == "__main__":
    main()
