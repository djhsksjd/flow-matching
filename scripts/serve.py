"""
Web-based Visualization Server

Start a web server to interactively visualize Flow Matching generation.

Usage:
    python -m scripts.serve
    python scripts/serve.py [--port PORT] [--host HOST]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
import base64
from io import BytesIO
from flask import Flask, render_template_string, jsonify, request
from flowmatching import UNet, FlowMatching
from flowmatching.utils import load_checkpoint
from flowmatching.config import MODEL_CONFIG, SAMPLE_CONFIG, PATHS
import numpy as np
from PIL import Image

app = Flask(__name__)
flow_matching = None
model_loaded = False


def tensor_to_base64(tensor):
    """Convert tensor to base64 encoded image."""
    img = tensor.cpu().numpy()
    if img.shape[0] == 3:  # RGB
        img = img.transpose(1, 2, 0)
        img = (img * 255).astype(np.uint8)
        img_pil = Image.fromarray(img)
    else:  # Grayscale
        img = img.squeeze()
        img = (img * 255).astype(np.uint8)
        img_pil = Image.fromarray(img, mode='L')
    
    buffered = BytesIO()
    img_pil.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def generate_steps(num_steps=100, num_samples=1):
    """Generate intermediate steps."""
    global flow_matching
    if flow_matching is None:
        return None
    
    image_size = SAMPLE_CONFIG['image_size']
    channels = MODEL_CONFIG['in_channels']
    
    flow_matching.model.eval()
    x = torch.randn(num_samples, channels, image_size[0], image_size[1], device=flow_matching.device)
    steps = []
    dt = 1.0 / num_steps
    
    with torch.no_grad():
        for i in range(num_steps + 1):
            t = torch.full((num_samples,), i * dt, device=flow_matching.device)
            v = flow_matching.model(x, t)
            if i < num_steps:
                x = x + dt * v
                x = torch.clamp(x, 0, 1)
            img_str = tensor_to_base64(x[0])
            steps.append({
                'step': i,
                'total_steps': num_steps,
                'image': img_str,
                'progress': (i / num_steps) * 100
            })
    
    return steps


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Flow Matching Generation Visualizer</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
        .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; text-align: center; }
        .controls { margin: 20px 0; padding: 20px; background: #f9f9f9; border-radius: 5px; }
        .control-group { margin: 15px 0; }
        label { display: inline-block; width: 150px; font-weight: bold; }
        input[type="number"] { width: 100px; padding: 5px; }
        button { background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; margin: 5px; }
        button:hover { background: #45a049; }
        button:disabled { background: #cccccc; cursor: not-allowed; }
        .status { margin: 10px 0; padding: 10px; border-radius: 5px; }
        .status.info { background: #e3f2fd; color: #1976d2; }
        .status.error { background: #ffebee; color: #c62828; }
        .status.success { background: #e8f5e9; color: #2e7d32; }
        #image-container { text-align: center; margin: 20px 0; min-height: 400px; display: flex; align-items: center; justify-content: center; }
        #generated-image { max-width: 100%; border: 2px solid #ddd; border-radius: 5px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .progress-bar { width: 100%; height: 30px; background: #e0e0e0; border-radius: 15px; overflow: hidden; margin: 10px 0; }
        .progress-fill { height: 100%; background: linear-gradient(90deg, #4CAF50, #8BC34A); transition: width 0.3s; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎨 Flow Matching Generation Visualizer</h1>
        <div class="controls">
            <div class="control-group">
                <label>Number of Steps:</label>
                <input type="number" id="numSteps" value="100" min="10" max="500" step="10">
                <small>(More steps = better quality, slower generation)</small>
            </div>
            <div class="control-group">
                <button id="generateBtn" onclick="generate()">Generate Image</button>
                <button id="animateBtn" onclick="animate()" disabled>Animate Process</button>
                <button onclick="stopAnimation()">Stop Animation</button>
            </div>
        </div>
        <div id="status"></div>
        <div class="progress-bar" id="progressBar" style="display: none;">
            <div class="progress-fill" id="progressFill" style="width: 0%">0%</div>
        </div>
        <div id="image-container">
            <img id="generated-image" style="display: none;" alt="Generated image">
        </div>
    </div>
    <script>
        let animationInterval = null;
        let currentSteps = [];
        let currentStepIndex = 0;
        function showStatus(message, type) {
            const statusDiv = document.getElementById('status');
            statusDiv.className = 'status ' + type;
            statusDiv.textContent = message;
        }
        function showProgress(percent) {
            const progressBar = document.getElementById('progressBar');
            const progressFill = document.getElementById('progressFill');
            progressBar.style.display = 'block';
            progressFill.style.width = percent + '%';
            progressFill.textContent = Math.round(percent) + '%';
        }
        function hideProgress() {
            document.getElementById('progressBar').style.display = 'none';
        }
        async function generate() {
            const numSteps = parseInt(document.getElementById('numSteps').value);
            const generateBtn = document.getElementById('generateBtn');
            const animateBtn = document.getElementById('animateBtn');
            generateBtn.disabled = true;
            animateBtn.disabled = true;
            showStatus('Generating image...', 'info');
            showProgress(0);
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({num_steps: numSteps})
                });
                const data = await response.json();
                if (data.success) {
                    currentSteps = data.steps;
                    document.getElementById('generated-image').src = 'data:image/png;base64,' + data.final_image;
                    document.getElementById('generated-image').style.display = 'block';
                    animateBtn.disabled = false;
                    showStatus('✅ Image generated successfully!', 'success');
                } else {
                    showStatus('❌ Error: ' + data.error, 'error');
                }
            } catch (error) {
                showStatus('❌ Error: ' + error.message, 'error');
            } finally {
                generateBtn.disabled = false;
                hideProgress();
            }
        }
        function animate() {
            if (currentSteps.length === 0) {
                showStatus('Please generate an image first', 'error');
                return;
            }
            const img = document.getElementById('generated-image');
            currentStepIndex = 0;
            animationInterval = setInterval(() => {
                if (currentStepIndex < currentSteps.length) {
                    const step = currentSteps[currentStepIndex];
                    img.src = 'data:image/png;base64,' + step.image;
                    showProgress(step.progress);
                    currentStepIndex++;
                } else {
                    stopAnimation();
                }
            }, 50);
            showStatus('Animating generation process...', 'info');
        }
        function stopAnimation() {
            if (animationInterval) {
                clearInterval(animationInterval);
                animationInterval = null;
            }
            if (currentSteps.length > 0) {
                const finalStep = currentSteps[currentSteps.length - 1];
                document.getElementById('generated-image').src = 'data:image/png;base64,' + finalStep.image;
                showProgress(100);
            }
        }
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/generate', methods=['POST'])
def generate():
    global flow_matching, model_loaded
    if not model_loaded:
        return jsonify({'success': False, 'error': 'Model not loaded. Please check server logs.'})
    
    try:
        data = request.json
        num_steps = data.get('num_steps', 100)
        steps = generate_steps(num_steps=num_steps, num_samples=1)
        
        if steps is None:
            return jsonify({'success': False, 'error': 'Failed to generate steps'})
        
        steps_data = [{
            'step': step['step'],
            'total_steps': step['total_steps'],
            'image': step['image'],
            'progress': step['progress']
        } for step in steps]
        
        final_image = steps_data[-1]['image']
        return jsonify({
            'success': True,
            'steps': steps_data,
            'final_image': final_image
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


def load_model(checkpoint_path=None):
    global flow_matching, model_loaded
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if checkpoint_path is None:
        results_dir = PATHS['results_dir']
        final_checkpoint = os.path.join(results_dir, 'checkpoint_final.pt')
        if os.path.exists(final_checkpoint):
            checkpoint_path = final_checkpoint
        else:
            checkpoints = [f for f in os.listdir(results_dir) if f.startswith('checkpoint_epoch_')]
            if checkpoints:
                epochs = [int(f.split('_')[2].split('.')[0]) for f in checkpoints]
                latest_idx = epochs.index(max(epochs))
                checkpoint_path = os.path.join(results_dir, checkpoints[latest_idx])
            else:
                raise FileNotFoundError(f"No checkpoint found in {results_dir}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    model = UNet(**MODEL_CONFIG)
    epoch, loss = load_checkpoint(checkpoint_path, model, device=device)
    print(f"✅ Loaded checkpoint from epoch {epoch}")
    
    flow_matching = FlowMatching(model, device=device)
    model_loaded = True
    print("✅ Model loaded and ready!")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Web-based Flow Matching visualizer')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the web server (default: 5000)')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to bind to (default: 127.0.0.1)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Flow Matching Web Visualizer")
    print("=" * 60)
    
    try:
        load_model(args.checkpoint)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    print(f"\n🌐 Starting web server on http://{args.host}:{args.port}")
    print("   Open this URL in your browser to visualize generation!")
    print("\n   Press Ctrl+C to stop the server")
    
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()
