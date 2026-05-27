import os
import subprocess
import yaml

def run_command(command, shell=True):
    """Función auxiliar para ejecutar comandos de consola de forma limpia."""
    print(f"Ejecutando: {command}")
    try:
        subprocess.run(command, shell=shell, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar el comando: {e}")
        raise e

def setup_environment():
    print("=== 1. Clonando repositorios y descargando scripts auxiliares ===")
    os.chdir("/root")
    
    # Clonar el repositorio principal de entrenamiento
    run_command("git clone https://github.com/MightyCrimsonX/LoRA_Easy_Training_Scripts.git")
    
    # Descargar los widgets y scripts complementarios
    run_command("wget -q https://raw.githubusercontent.com/MightyCrimsonX/MightyCrimson_Lora_Scripts/refs/heads/Anima_Modal/Notebooks_scripts/sample_prompt_widget.py")
    run_command("wget -q https://raw.githubusercontent.com/MightyCrimsonX/Notebook_Scripts/refs/heads/Dev/scripts/download_magic.py")
    
    print("\n=== 2. Instalando dependencias iniciales ===")
    # Instalar gestor de paquetes uv y aria2/gdown para descargas
    run_command("pip install -q uv")
    run_command("uv pip install --system aria2 gdown pyyaml")
    
    # Actualizar submódulos del repositorio
    os.chdir("/root/LoRA_Easy_Training_Scripts")
    run_command("git submodule update --init --recursive")
    
    # Instalar requerimientos del repositorio base
    run_command("uv pip install --system -r requirements.txt")
    
    print("\n=== 3. Instalando dependencias de optimización y PyTorch ===")
    os.chdir("/root/LoRA_Easy_Training_Scripts/backend/sd_scripts")
    
    # Instalación de extensiones de tipado y entorno de ejecución Torch cu128
    run_command("uv pip install --system -U typing-extensions~=4.15.0")
    run_command("uv pip install --system torch~=2.7.1 torchvision~=0.22.1 --index-url https://download.pytorch.org/whl/cu128")
    
    # Forzar la reinstalación de RamTorch sin dependencias
    run_command("uv pip install --system -U --no-deps --force-reinstall git+https://github.com/67372a/RamTorch")
    
    # Instalar Xformers y Torchao optimizados para CUDA 12.8
    run_command("uv pip install --system -U --no-deps xformers==0.0.31.post1 --index-url https://download.pytorch.org/whl/cu128")
    run_command("uv pip install --system -U --no-deps torchao~=0.12.0 --index-url https://download.pytorch.org/whl/cu128")
    
    # Completar la instalación de dependencias de sd_scripts y schedulers personalizados
    run_command("uv pip install --system -U -r requirements.txt")
    run_command("uv pip install --system -U ../custom_scheduler/.")
    run_command("uv pip install --system -U -r ../requirements.txt")
    
    # Instalar la rama de desarrollo de LyCORIS
    run_command("uv pip install --system -U --force-reinstall --no-deps git+https://github.com/67372a/LyCORIS@dev")


def create_accelerate_config():
    print("\n=== 4. Configurando Accelerate de forma silenciosa ===")
    config_path = os.path.expanduser("~/.cache/huggingface/accelerate/default_config.yaml")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    # Estructura de configuración óptima para una sola GPU
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
    print(f"Configuración guardada con éxito en: {config_path}")


def download_models():
    print("\n=== 5. Descargando Modelos Base (Anima) ===")
    models_dir = "/root/models"
    os.makedirs(models_dir, exist_ok=True)
    os.chdir(models_dir)
    
    urls = [
        "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/diffusion_models/anima-base-v1.0.safetensors",
        "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/vae/qwen_image_vae.safetensors",
        "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/text_encoders/qwen_3_06b_base.safetensors"
    ]
    
    # Descarga multi-hilo forzando el nombre correcto mediante el parámetro --out (-o)
    for url in urls:
        filename = url.split("/")[-1]
        print(f"Descargando archivo: {filename}...")
        # El argumento -o asegura que conserve el nombre 'anima-base-v1.0.safetensors', etc.
        run_command(f"aria2c -x 16 -s 16 -d {models_dir} -o {filename} '{url}'")


if __name__ == "__main__":
    print("Iniciando script de instalación y descarga...")
    setup_environment()
    create_accelerate_config()
    download_models()
    print("\n=============================================")
    print(" INSTALACIÓN Y DESCARGAS COMPLETADAS CON ÉXITO")
    print("=============================================")