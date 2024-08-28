import os
import subprocess
import sys
import venv
import torch
import argparse
import threading
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from tqdm import tqdm


def run_command(command):
    result = subprocess.run(command, shell=True, executable='/bin/bash', capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error executing command: {command}")
        print(result.stderr)
        sys.exit(1)
    return result.stdout

if not os.path.exists("ai-toolkit"):
    run_command("git clone https://github.com/ostris/ai-toolkit.git")
    print("Installed the ai-toolkit....")
    os.chdir("ai-toolkit")
    run_command("git submodule update --init --recursive")
    print("Installed the submodules dependencies....")
else:
    print("ai-toolkit directory already exists. Skipping clone.")
    os.chdir("ai-toolkit")

venv.create("venv", with_pip=True)
print("Initialized environment....")

activate_script = os.path.join("venv", "bin", "activate")
subprocess.run(["bash", "-c", f"source {activate_script}"], check=True)
setup_commands = [
    "pip install --upgrade pip",
    "pip install -r requirements.txt",
    "pip install torch",
    "pip install flash_attn einops timm"
]
for command in setup_commands:
    run_command(command)

print("Installed the ai-toolkit dependencies....")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print("Initializing model and processor...")
model_id = 'microsoft/Florence-2-large'
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype='auto').eval().to(device)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

print("Model and processor loaded")
def generate_caption(image_path):
    image = Image.open(image_path)
    prompt = "Describe this image in detail."
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
    
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        do_sample=False
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task="<MORE_DETAILED_CAPTION>", image_size=(image.width, image.height))
    
    return parsed_answer["<MORE_DETAILED_CAPTION>"]

def process_directory(directory_path):
    image_files = [f for f in os.listdir(directory_path) if f.lower().endswith(('.jpeg', '.jpg', '.png', '.webp'))]
    total_images = len(image_files)
    
    print(f"Found {total_images} images to process.")
    
    for i, filename in enumerate(tqdm(image_files, desc="Processing images", unit="image")):
        image_path = os.path.join(directory_path, filename)
        try:
            caption = generate_caption(image_path)
            
            # Create text file name
            txt_filename = os.path.splitext(filename)[0] + '.txt'
            txt_path = os.path.join(directory_path, txt_filename)
            
            # Save caption to text file
            with open(txt_path, 'w') as f:
                f.write(caption)
            
            tqdm.write(f"Processed {filename} ({i+1}/{total_images})")
        except Exception as e:
            tqdm.write(f"Error processing {filename}: {str(e)}")


download_complete = threading.Event()

def download_file(url, filename):
    print(f"Starting download of {filename} in the background...")
    response = requests.get(url, stream=True)
    with open(filename, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
    download_complete.set()
    print(f"Download of {filename} completed.")
    

def check_download_status():
    if download_complete.is_set():
        print("Download completed.")
    else:
        print("Download still in progress...")


url = "https://huggingface.co/ostris/FLUX.1-schnell-training-adapter/resolve/main/pytorch_lora_weights.safetensors?download=true"
filename = "pytorch_lora_weights.safetensors"
download_thread = threading.Thread(target=download_file, args=(url, filename))
download_thread.start()

print("The model weights are being downloaded in the background.")
print("You can continue using your system. The download will complete automatically.")
print("To check the download status, you can run this script again with the --check-download flag.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate captions for images in a directory.")
    parser.add_argument("directory_path", type=str, help="Path to the directory containing images")
    args = parser.parse_args()

    if not os.path.isdir(args.directory_path):
        print(f"Error: {args.directory_path} is not a valid directory.")
        sys.exit(1)

    print("Starting image captioning process...")
    print(f"Saving to this Directory: {args.directory_path}")
    process_directory(args.directory_path)
    print("Image captioning process completed. Now run the run.py")