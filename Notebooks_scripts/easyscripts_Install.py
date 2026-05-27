import os
import subprocess
import yaml
import modal

# 1. Definimos la imagen de Modal con todas las dependencias preinstaladas.
# Esto evita que el código de instalación ensucie la ejecución del script.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "aria2")
    # Instala uv para gestionar paquetes de forma ultra rápida
    .pip_install("uv")
    # Ejecuta las instalaciones tal como estaban en tu cuaderno
    .run_commands(
        "uv pip install --system gdown pyyaml",
        "git clone https://github.com/MightyCrimsonX/LoRA_Easy_Training_Scripts.git /root/LoRA_Easy_Training_Scripts",
        "wget -q https://raw.githubusercontent.com/MightyCrimsonX/MightyCrimson_Lora_Scripts/refs/heads/Anima_Modal/Notebooks_scripts/sample_prompt_widget.py -O /root/sample_prompt_widget.py",
        "wget -q https://raw.githubusercontent.com/MightyCrimsonX/Notebook_Scripts/refs/heads/Dev/scripts/download_magic.py -O /root/download_magic.py",
    )
    .run_commands(
        "cd /root/LoRA_Easy_Training_Scripts && git submodule update --init --recursive",
        "cd /root/LoRA_Easy_Training_Scripts && uv pip install --system -r requirements.txt",
    )
    .run_commands(
        "uv pip install --system -U 'typing-extensions>=4.15.0'",
        "uv pip install --system torch torchvision --index-url https://download.pytorch.org/whl/cu128",
        "uv pip install --system -U --no-deps --force-reinstall git+https://github.com/67372a/RamTorch",
        "uv pip install --system -U --no-deps xformers --index-url https://download.pytorch.org/whl/cu128",
        "uv pip install --system -U --no-deps torchao --index-url https://download.pytorch.org/whl/cu128",
    )
    .run_commands(
        "cd /root/LoRA_Easy_Training_Scripts/backend/sd_scripts && uv pip install --system -U -r requirements.txt",
        "cd /root/LoRA_Easy_Training_Scripts/backend/sd_scripts && uv pip install --system -U ../custom_scheduler/.",
        "cd /root/LoRA_Easy_Training_Scripts/backend/sd_scripts && uv pip install --system -U -r ../requirements.txt",
        "uv pip install --system -U --force-reinstall --no-deps git+https://github.com/67372a/LyCORIS@dev",
    )
)

app = modal.App("lora-training-setup", image=image)

# 2. Función auxiliar para escribir la configuración de Accelerate
def create_accelerate_config():
    config_path = os.path.expanduser("~/.cache/huggingface/accelerate/default_config.yaml")
    if not os.path.exists(config_path):
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        accelerate_config = {
            "compute_environment": "LOCAL_MACHINE",
            "distributed_type": "NO",
            "downcast_bf16": "no",
            "gpu_ids": "all",
            "machine_rank": 0,
            "main_training_function": "main",
            "mixed_precision": "fp16",
            "num_machines": 1,
            "num_processes": 1,
            "rdzv_backend": "static",
            "same_network": True,
            "tpu_env": [],
            "tpu_use_cluster": False,
            "tpu_use_sudo": False,
            "use_cpu": False
        }
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(accelerate_config, f, default_flow_style=False)
        print("Configuración de Accelerate generada de forma silenciosa.")

# 3. Función principal que se ejecuta en la nube de Modal
@app.function(gpu="A10G", timeout=3600) # Puedes cambiar el tipo de GPU según tus necesidades
def run_setup():
    # Crear configuración
    create_accelerate_config()
    
    # Crear directorio de modelos
    os.makedirs("/root/models", exist_ok=True)
    
    # Nota sobre descargas de HuggingFace:
    # Las funciones mágicas como '%download' son exclusivas de Jupyter.
    # En un script de Python estándar, se utilizan herramientas como `wget` o `aria2c` vía subprocess.
    print("Descargando modelos base...")
    
    urls = [
        "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/diffusion_models/anima-base-v1.0.safetensors",
        "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/vae/qwen_image_vae.safetensors",
        "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/text_encoders/qwen_3_06b_base.safetensors"
    ]
    
    for url in urls:
        filename = url.split("/")[-1]
        dest_path = f"/root/models/{filename}"
        if not os.path.exists(dest_path):
            print(f"Descargando {filename}...")
            # Usamos aria2c para descargas rápidas en paralelo
            subprocess.run(["aria2c", "-x", "16", "-s", "16", "-d", "/root/models", url], check=True)
            
    print("Instalación y preparación completadas exitosamente.")

# Punto de entrada local
if __name__ == "__main__":
    with app.run():
        run_setup.remote()