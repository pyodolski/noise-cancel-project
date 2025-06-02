import os
import shutil
import torch
import torchaudio
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from demucs.pretrained import get_model
from tempfile import TemporaryDirectory

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_wrapper = get_model('htdemucs').to(device)
model = model_wrapper.models[0].to(device)
model.load_state_dict(torch.load('outputs/epoch_30_data_200.th', map_location=device))
model.eval()

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <html>
        <body>
            <h2>Noise Removal Upload</h2>
            <form action="/upload/" enctype="multipart/form-data" method="post">
                <input name="noisy_file" type="file">
                <input type="submit">
            </form>
        </body>
    </html>
    """

@app.post("/upload/")
async def upload_wav(noisy_file: UploadFile = File(...)):
    with TemporaryDirectory() as tmpdir:
        noisy_path = os.path.join("static", noisy_file.filename)
        output_path = os.path.join("static", f"denoised_{noisy_file.filename}")

        with open(noisy_path, 'wb') as f:
            shutil.copyfileobj(noisy_file.file, f)

        noisy, sr = torchaudio.load(noisy_path)

        if noisy.shape[0] == 1:
            noisy = noisy.repeat(2, 1)

        noisy_input = noisy.unsqueeze(0).to(device)

        with torch.no_grad():
            estimate = model(noisy_input).cpu()

        if estimate.dim() == 4:
            estimate_to_save = estimate[0].sum(dim=0)
        elif estimate.dim() == 3:
            estimate_to_save = estimate[0]
        else:
            estimate_to_save = estimate

        torchaudio.save(output_path, estimate_to_save, sr)

        html_content = f"""
        <html>
            <body>
                <h2>Noise Removal Results</h2>
                <p>Original File:</p>
                <audio controls>
                    <source src="/static/{noisy_file.filename}" type="audio/wav">
                    Your browser does not support the audio element.
                </audio>
                <p>Denoised File:</p>
                <audio controls>
                    <source src="/static/denoised_{noisy_file.filename}" type="audio/wav">
                    Your browser does not support the audio element.
                </audio>
                <br><br>
                <a href="/">Upload Another File</a>
            </body>
        </html>
        """
        return HTMLResponse(content=html_content)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)