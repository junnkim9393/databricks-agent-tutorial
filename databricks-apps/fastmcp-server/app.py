import os

import uvicorn
from fastapi import FastAPI

from server import mcp

app = FastAPI(title="MCP Tutorial Server")


@app.get("/health")
def health():
    return {"status": "ok"}


app.mount("/", mcp.http_app(transport="sse"))


if __name__ == "__main__":
    port = int(os.environ.get("DATABRICKS_APP_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
