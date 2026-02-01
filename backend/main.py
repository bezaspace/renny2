import base64
import json
import os
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from google.adk.artifacts import InMemoryArtifactService
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel

from strategy_pipeline import build_agent_pipeline
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    message: str
    session_id: str | None = None

MODEL_NAME = os.getenv("LITELLM_MODEL")

if not MODEL_NAME:
    raise RuntimeError(
        "Missing LITELLM_MODEL. Set it in backend/.env (e.g. openai/gpt-4o)."
    )

APP_NAME = "trading_copilot_app"
USER_ID = "web_user"

agent = build_agent_pipeline(LiteLlm(model=MODEL_NAME))

session_service = InMemorySessionService()
artifact_service = InMemoryArtifactService()
runner = Runner(
    agent=agent,
    app_name=APP_NAME,
    session_service=session_service,
    artifact_service=artifact_service,
)

active_sessions: set[str] = set()


@app.post("/chat")
async def chat(chat_message: ChatMessage):
    """Non-streaming chat endpoint (kept for backward compatibility)."""
    user_message = chat_message.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    session_id = chat_message.session_id or str(uuid.uuid4())
    if session_id not in active_sessions:
        await session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=session_id,
        )
        active_sessions.add(session_id)

    content = types.Content(role="user", parts=[types.Part(text=user_message)])
    final_text = None
    tool_summary = None

    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=content,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.function_response:
                    response = part.function_response.response or {}
                    if isinstance(response, dict):
                        tool_summary = response.get("summary") or tool_summary
        if event.is_final_response() and event.content and event.content.parts:
            final_text = event.content.parts[0].text

    if not final_text:
        if tool_summary:
            return {"response": tool_summary, "session_id": session_id}
        raise HTTPException(status_code=502, detail="No response from agent.")

    return {"response": final_text, "session_id": session_id}


@app.post("/chat/stream")
async def chat_stream(chat_message: ChatMessage):
    """Streaming chat endpoint using Server-Sent Events."""
    user_message = chat_message.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    session_id = chat_message.session_id or str(uuid.uuid4())
    if session_id not in active_sessions:
        await session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=session_id,
        )
        active_sessions.add(session_id)

    async def generate_stream():
        content = types.Content(role="user", parts=[types.Part(text=user_message)])
        last_text = ""
        run_config = RunConfig(
            streaming_mode=StreamingMode.SSE,
            response_modalities=["TEXT"],
        )
        
        async for event in runner.run_async(
            user_id=USER_ID,
            session_id=session_id,
            new_message=content,
            run_config=run_config,
        ):
            # Stream partial content as it arrives
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.function_response:
                        response = part.function_response.response or {}
                        if isinstance(response, dict):
                            artifacts_payload = []
                            artifacts = response.get("artifacts") or []
                            for artifact in artifacts:
                                filename = artifact.get("filename")
                                if not filename:
                                    continue
                                loaded = await artifact_service.load_artifact(
                                    app_name=APP_NAME,
                                    user_id=USER_ID,
                                    session_id=session_id,
                                    filename=filename,
                                )
                                if not loaded or not loaded.inline_data:
                                    continue
                                data_b64 = base64.b64encode(loaded.inline_data.data).decode("ascii")
                                artifacts_payload.append(
                                    {
                                        "name": filename,
                                        "label": artifact.get("label") or filename,
                                        "mime_type": loaded.inline_data.mime_type,
                                        "data": data_b64,
                                    }
                                )
                            tool_data = {
                                "type": "tool_result",
                                "step": response.get("step"),
                                "summary": response.get("summary"),
                                "artifacts": artifacts_payload,
                                "metrics": response.get("metrics"),
                                "session_id": session_id,
                            }
                            yield f"data: {json.dumps(tool_data)}\n\n"
                    if part.text:
                        text = part.text
                        if text.startswith(last_text):
                            delta = text[len(last_text):]
                        else:
                            delta = text
                        if not delta:
                            continue
                        last_text = text
                        # Send each chunk as a Server-Sent Event
                        data = json.dumps({"type": "text", "chunk": delta, "session_id": session_id})
                        yield f"data: {data}\n\n"
            
            # Send final response marker
            if event.is_final_response():
                yield "data: [DONE]\n\n"
                break

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
