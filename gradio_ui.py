import subprocess
import time

subprocess.Popen([
    "uvicorn",
    "server.app:app",
    "--host", "0.0.0.0",
    "--port", "8000"
])

# small delay to let server boot
time.sleep(2)

# 👇 rest of your code
import gradio as gr
import asyncio
from inference import run_inference


def run_simulation(task):
    try:
        result = asyncio.run(run_inference(task))

        if "error" in result:
            return f"❌ Error: {result['error']}"

        return f"""
🔥 Task: {result['task']}

✅ Success: {result['success']}
📊 Score: {result['score']:.3f}
🔁 Steps: {result['steps']}

🎯 Rewards:
{result['rewards']}
"""
    except Exception as e:
        return f"❌ Failed: {str(e)}"


demo = gr.Interface(
    fn=run_simulation,
    inputs=gr.Dropdown([1, 2, 3], label="Select Task", value=1),
    outputs=gr.Textbox(label="Output"),
    title="Wildfire Simulation Enviroment",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)