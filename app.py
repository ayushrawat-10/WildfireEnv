import os
import uvicorn

if __name__ == "__main__":
    # If Hugging Face forces app.py as the entrypoint (Gradio Space fallback),
    # this will route it to start the exact same graphical OpenEnv FastAPI backend!
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)
