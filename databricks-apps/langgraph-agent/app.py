import logging
import os
import sys
from contextlib import asynccontextmanager

import uvicorn
from databricks.sdk import WorkspaceClient
from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from agent import build_agent

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL")
UC_FUNCTIONS = [f.strip() for f in os.environ.get("UC_FUNCTIONS", "").split(",") if f.strip()]

agent = None
startup_error = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent, startup_error
    try:
        logger.info("Initializing agent...")
        wc = WorkspaceClient()
        tools = []

        if UC_FUNCTIONS:
            from databricks_langchain import UCFunctionToolkit
            from unitycatalog.ai.core.databricks import DatabricksFunctionClient
            logger.info(f"Loading UC functions: {UC_FUNCTIONS}")
            uc_client = DatabricksFunctionClient(client=wc)
            toolkit = UCFunctionToolkit(function_names=UC_FUNCTIONS, client=uc_client)
            tools.extend(toolkit.tools)

        if MCP_SERVER_URL:
            from langchain_mcp_adapters.client import MultiServerMCPClient
            logger.info(f"Connecting to MCP server: {MCP_SERVER_URL}")
            auth_headers = wc.config.authenticate()
            authorization = auth_headers.get("Authorization", "")
            logger.info(f"Auth header non-empty: {bool(authorization)}, client_id set: {bool(os.environ.get('DATABRICKS_CLIENT_ID'))}")
            mcp_client = MultiServerMCPClient(
                {
                    "databricks_mcp": {
                        "url": MCP_SERVER_URL,
                        "transport": "sse",
                        "headers": {"Authorization": authorization},
                    }
                }
            )
            tools.extend(await mcp_client.get_tools())

        agent = build_agent(tools)
        logger.info("Agent initialized successfully.")
    except Exception as e:
        import traceback
        startup_error = traceback.format_exc()
        print("STARTUP ERROR:", startup_error, flush=True)
    yield


app = FastAPI(title="LangGraph Agent", lifespan=lifespan)


class QueryRequest(BaseModel):
    message: str


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/debug")
def debug():
    return {"agent_ready": agent is not None, "startup_error": startup_error}


@app.post("/invoke")
async def invoke(request: QueryRequest):
    if agent is None:
        raise HTTPException(status_code=503, detail=startup_error or "Agent failed to initialize")
    try:
        result = await agent.ainvoke({"messages": [HumanMessage(content=request.message)]})
        messages = [
            {"type": m.__class__.__name__, "content": m.content, "tool_calls": getattr(m, "tool_calls", [])}
            for m in result["messages"]
        ]
        return {"response": result["messages"][-1].content, "messages": messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("DATABRICKS_APP_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
