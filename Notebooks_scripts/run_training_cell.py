"""Notebook entry-point for launching the Anima LoRA training run.

This script is adapted from the original SDXL training cell to support Anima
(DiT-based architecture with Qwen3 text encoder and Qwen-Image VAE).
It can be executed with ``exec(open(...).read(), globals())`` while keeping the
notebook clean.  It may still rely on global variables defined by previous cells.
"""
import os, re, sys, toml
from pathlib import Path
from time import time
import time
from IPython.display import Markdown, display, HTML, clear_output
from huggingface_hub.utils import disable_progress_bars
import logging

from IPython import get_ipython
import subprocess


disable_progress_bars()
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("ACCELERATE_DISABLE_RICH_PROGRESS", "1")
os.environ.setdefault("ACCELERATE_DISABLE_PROGRESS_BAR", "1")
os.environ.setdefault("TQDM_MININTERVAL", "2")
os.environ.setdefault("TQDM_NOPOS", "1")
logging.getLogger("accelerate").setLevel(logging.WARNING)
logging.getLogger("accelerate.tracking").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.WARNING)

logging.getLogger("tqdm").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.WARNING)

def _run_cmd(command: str) -> None:
    ip = get_ipython()
    if ip is not None:
        ip.system(command)
    else:
        subprocess.run(command, shell=True, check=True)


root_dir = "/root"
trainer_dir = os.path.join(root_dir, "LoRA_Easy_Training_scripts_Backend")
kohya_dir = os.path.join(root_dir, "sd-scripts")
models_dir = "/root/models"
downloads_dir = os.path.join(root_dir, "downloads")
custom_optimizer_path = os.path.join(trainer_dir, "custom_scheduler")
if custom_optimizer_path not in sys.path:
  sys.path.append(custom_optimizer_path)
os.environ["PYTHONPATH"] = custom_optimizer_path + os.pathsep + os.environ.get("PYTHONPATH", "")

# Lightning notebooks run continuously; automatic shutdown is not managed here.
print("🔵 Lightning environment detectado. Detén el cuaderno manualmente cuando termines.")

# These carry information from past executions
if "model_url" in globals():
  old_model_url = model_url
else:
  old_model_url = None
if "dependencies_installed" not in globals():
  dependencies_installed = False
if "model_file" not in globals():
  model_file = None

# These may be set by other cells, some are legacy
if "custom_dataset" not in globals():
  custom_dataset = None
if "override_dataset_config_file" not in globals():
  override_dataset_config_file = None
if "override_config_file" not in globals():
  override_config_file = None

COMMIT = "fa2427c6b468231e8e270e40fe72add780118dbe"
LOWRAM = False
LOAD_TRUNCATED_IMAGES = True
BETTER_EPOCH_NAMES = True
FIX_DIFFUSERS = False
FIX_WANDB_WARNING = True

#@title ## 🚩 Start Here (Anima LoRA Training)

#@markdown ### ▶️ Setup
#@markdown El nombre de tu proyecto será el mismo que el de la carpeta que contiene tus imágenes. No se permiten espacios, puedes usar `guión bajo` si el nombre es muy largo.
project_name_param = " " #@param {type:"string"}
project_name = globals().get("project_name", project_name_param).strip()
#@markdown La estructura de carpetas no importa y es puramente por comodidad. Asegúrate de elegir siempre el mismo.  Me gusta organizar por proyecto.
folder_structure = "Organize by project (lora_projects/project_name/dataset)" #@param ["Organize by category (lora_training/datasets/project_name)", "Organize by project (lora_projects/project_name/dataset)"]

#@markdown #### Modelo Anima
#@markdown Selecciona el modelo Anima DiT base. También puedes pegar un enlace de descarga o proporcionar una ruta local.
training_model_param = "Anima-Preview" # @param ["Anima-Preview"]
training_model = globals().get("training_model", training_model_param)
optional_custom_training_model_param = "" #@param {type:"string"}
optional_custom_training_model = str(globals().get("optional_custom_training_model", optional_custom_training_model_param)).strip()
#@markdown Activa esta opción para utilizar el modelo personalizado descargado.
use_optional_custom_training_model_param = False #@param {type:"boolean"}
use_optional_custom_training_model = bool(globals().get("use_optional_custom_training_model", use_optional_custom_training_model_param))

#@markdown #### Rutas de componentes Anima
#@markdown Qwen3-0.6B y Qwen-Image VAE se descargan automáticamente en la carpeta de modelos.
qwen3_path = os.path.join(models_dir, "qwen_3_06b_base.safetensors")
anima_vae_path = os.path.join(models_dir, "qwen_image_vae.safetensors")

# URLs de descarga para componentes Anima
qwen3_url = "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/text_encoders/qwen_3_06b_base.safetensors"
anima_vae_url = "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/vae/qwen_image_vae.safetensors"

#@markdown #### Rutas opcionales de componentes Anima
#@markdown Ruta al LLM Adapter (si no está incluido en el modelo DiT).
llm_adapter_path_param = "" #@param {type:"string"}
llm_adapter_path = str(globals().get("llm_adapter_path", llm_adapter_path_param)).strip()
#@markdown Ruta al T5 Tokenizer (si se desea usar uno distinto al incluido en configs/t5_old/).
t5_tokenizer_path_param = "" #@param {type:"string"}
t5_tokenizer_path = str(globals().get("t5_tokenizer_path", t5_tokenizer_path_param)).strip()

#@markdown Utilice wandb si desea visualizar el progreso de su entrenamiento a lo largo del tiempo.
wandb_key = "" #@param {type:"string"}

custom_model_selected = use_optional_custom_training_model and len(optional_custom_training_model) > 0

# --- Anima Model URLs ---
if custom_model_selected:
  model_url = optional_custom_training_model
elif "Anima-Preview" in training_model:
  model_url = "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/diffusion_models/anima-preview2.safetensors"
  model_file = os.path.join(models_dir, "anima_preview_dit.safetensors")
else:
  # Default fallback to Anima-Preview
  model_url = "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/diffusion_models/anima-preview2.safetensors"
  model_file = os.path.join(models_dir, "anima_preview_dit.safetensors")

# The VAE path is passed directly as --vae argument
vae_file = anima_vae_path

model_url = model_url.strip()

#@markdown ### ▶️ Processing
#@markdown Por defecto la resolución para Anima es 1024. Otras resoluciones posibles son 896 o 768.
resolution_param = 1024 #@param {type:"dropdown", min:768, max:1536, step:128}
resolution = globals().get("resolution", resolution_param)
#@markdown Activa `Flip Aug`si tu dataset es pequeño, volteará tus imágenes (modo espejo).
flip_aug = False #@param {type:"boolean"}
caption_extension = ".txt" # @param [".txt",".caption"]
#@markdown Mezcla etiquetas, mejora el aprendizaje. Una etiqueta de activación va al comienzo de cada archivo de texto y no se mezclará.
shuffle_tags = True #@param {type:"boolean"}
shuffle_caption = shuffle_tags
activation_tags = "1" #@param [0,1,2,3]
keep_tokens = int(activation_tags)

#@markdown ### ▶️ Steps
#@markdown Tus imágenes se repetirán esta cantidad de veces durante el entrenamiento.
num_repeats_param = 2 #@param {type:"number"}
num_repeats = globals().get("num_repeats", num_repeats_param)
#@markdown Elige cuánto tiempo quieres entrenar.
preferred_unit = "Epochs" #@param ["Epochs", "Steps"]
how_many_param = 40 #@param {type:"number"}
how_many = globals().get("how_many", how_many_param)
max_train_epochs = how_many if preferred_unit == "Epochs" else None
max_train_steps = how_many if preferred_unit == "Steps" else None
#@markdown Guardar más épocas te permitirá comparar el progreso de tu LoRA.
save_every_n_epochs = 1 #@param {type:"number"}
keep_only_last_n_epochs = 5 #@param {type:"number"}
if not save_every_n_epochs:
  save_every_n_epochs = max_train_epochs
if not keep_only_last_n_epochs:
  keep_only_last_n_epochs = max_train_epochs

#@markdown ### ▶️ Learning
#@markdown La tasa de aprendizaje para Anima. Valor recomendado: 1e-4 (para alpha=1.0 por defecto).
unet_lr_param = 1e-4 #@param {type:"number"}
unet_lr = globals().get("unet_lr", unet_lr_param)
#@markdown Learning rate del text encoder Qwen3. Se recomienda la mitad del unet_lr o menos. Ponlo en 0 para no entrenar el text encoder.
text_encoder_lr_param = 5e-5 #@param {type:"number"}
text_encoder_lr = globals().get("text_encoder_lr", text_encoder_lr_param)
#@markdown El scheduler es el algoritmo que guía la tasa de aprendizaje.
lr_scheduler_param = "constant_with_warmup" # @param ["constant","cosine","cosine_with_restarts","constant_with_warmup","linear","polynomial","rex"]
lr_scheduler = globals().get("lr_scheduler", lr_scheduler_param)
lr_scheduler_number = 0 #@param {type:"number"}
#@markdown Pasos de warmup como proporción del total.
lr_warmup_ratio = 0.05 #@param {type:"slider", min:0.0, max:0.2, step:0.01}
lr_warmup_steps = 100 #@param {type:"number"}
#@markdown `ip_noise_gamma` ajusta el ruido aleatorio. Nota: min_snr_gamma NO es compatible con Anima (usa Rectified Flow).
ip_noise_gamma_enabled = True #@param {type:"boolean"}
ip_noise_gamma = 0.05 #@param {type:"slider", min:0.05, max:0.1, step:0.01}

#@markdown ### ▶️ Text Encoder LoRA
#@markdown Activa `network_train_unet_only` para entrenar SOLO el DiT (sin LoRA del text encoder Qwen3). Recomendado activar si usas `cache_text_encoder_outputs`.
network_train_unet_only_param = True #@param {type:"boolean"}
network_train_unet_only = bool(globals().get("network_train_unet_only", network_train_unet_only_param))

#@markdown ### ▶️ Structure (Anima LoRA)
#@markdown Anima usa `networks.lora_anima` como módulo de red. LoCon no está disponible para Anima.
#@markdown A continuación se muestran valores recomendados:
#@markdown | Tipo | network_dim | network_alpha |
#@markdown | :---: | :---: | :---: |
#@markdown | Personaje LoRA | 8 | 4 |
#@markdown | Estilo LoRA | 16 | 8 |

network_dim_param = 8 #@param {type:"number", min:1, max:128, step:1}
network_dim = globals().get("network_dim", network_dim_param)
network_alpha_param = 4 #@param {type:"number", min:1, max:128, step:1}
network_alpha = globals().get("network_alpha", network_alpha_param)

# Anima uses networks.lora_anima - LoCon is not applicable
network_module = "networks.lora_anima"
network_args = None

#@markdown ### ▶️ Anima-Specific Parameters
#@markdown Método de muestreo de timesteps. `sigmoid` es el valor por defecto y funciona bien en general.
timestep_sampling_param = "sigmoid" #@param ["sigma", "uniform", "sigmoid", "shift", "flux_shift"]
timestep_sampling = globals().get("timestep_sampling", timestep_sampling_param)
#@markdown Shift para la distribución de timesteps en Rectified Flow. Solo aplica cuando timestep_sampling='shift'.
discrete_flow_shift_param = 1.0 #@param {type:"number"}
discrete_flow_shift = globals().get("discrete_flow_shift", discrete_flow_shift_param)
#@markdown Factor de escala para sigmoid/shift/flux_shift timestep sampling.
sigmoid_scale_param = 1.0 #@param {type:"number"}
sigmoid_scale = globals().get("sigmoid_scale", sigmoid_scale_param)
#@markdown Longitud máxima de tokens para Qwen3.
qwen3_max_token_length_param = 512 #@param {type:"number"}
qwen3_max_token_length = globals().get("qwen3_max_token_length", qwen3_max_token_length_param)
#@markdown Longitud máxima de tokens para T5.
t5_max_token_length_param = 512 #@param {type:"number"}
t5_max_token_length = globals().get("t5_max_token_length", t5_max_token_length_param)
#@markdown Esquema de ponderación de pérdida por timestep.
weighting_scheme_param = "uniform" #@param ["uniform", "sigma_sqrt", "cosmap", "none"]
weighting_scheme = globals().get("weighting_scheme", weighting_scheme_param)

#@markdown ### ▶️ Anima Memory Optimization
#@markdown Número de bloques Transformer para intercambiar entre CPU y GPU. Más bloques reducen VRAM pero ralentizan. Máx 26 para Anima-Preview (28 bloques).
blocks_to_swap_param = 0 #@param {type:"number"}
blocks_to_swap = globals().get("blocks_to_swap", blocks_to_swap_param)
#@markdown Chunk size para Qwen-Image VAE. Reduce VRAM a costa de velocidad.
vae_chunk_size_param = 64 #@param {type:"number"}
vae_chunk_size = globals().get("vae_chunk_size", vae_chunk_size_param)
#@markdown Desactivar caché interno del VAE para reducir VRAM.
vae_disable_cache_param = True #@param {type:"boolean"}
vae_disable_cache = globals().get("vae_disable_cache", vae_disable_cache_param)
#@markdown Offload de checkpoints de activación a CPU (async, más rápido que cpu_offload_checkpointing). No se puede usar con blocks_to_swap.
unsloth_offload_checkpointing_param = False #@param {type:"boolean"}
unsloth_offload_checkpointing = globals().get("unsloth_offload_checkpointing", unsloth_offload_checkpointing_param)

#@markdown ### ▶️ LLM Adapter & Regex Module Control
#@markdown Activa para aplicar LoRA también al LLM Adapter.
train_llm_adapter_param = False #@param {type:"boolean"}
train_llm_adapter = globals().get("train_llm_adapter", train_llm_adapter_param)
#@markdown Patrones de exclusión de módulos (regex, separados por comas). Dejar vacío para usar el patrón por defecto.
exclude_patterns_param = "" #@param {type:"string"}
exclude_patterns = str(globals().get("exclude_patterns", exclude_patterns_param)).strip()
#@markdown Patrones de inclusión forzada de módulos (regex, separados por comas). Dejar vacío para no forzar ninguno.
include_patterns_param = "" #@param {type:"string"}
include_patterns = str(globals().get("include_patterns", include_patterns_param)).strip()
#@markdown Dims por módulo (regex). Formato: `.*self_attn.*=8,.*cross_attn.*=4`. Dejar vacío para usar network_dim global.
network_reg_dims_param = "" #@param {type:"string"}
network_reg_dims = str(globals().get("network_reg_dims", network_reg_dims_param)).strip()
#@markdown Learning rates por módulo (regex). Formato: `.*self_attn.*=1e-4,.*cross_attn.*=5e-5`. Dejar vacío para usar lr global.
network_reg_lrs_param = "" #@param {type:"string"}
network_reg_lrs = str(globals().get("network_reg_lrs", network_reg_lrs_param)).strip()

#@markdown ### ▶️ Training
#@markdown Ajuste estos parámetros según la configuración de su entorno.
train_batch_size_param = 4 #@param {type:"slider", min:1, max:16, step:1}
train_batch_size = globals().get("train_batch_size", train_batch_size_param)
#@markdown Implementación de atención a usar. `torch` es el valor por defecto.
attn_mode_param = "torch" #@param ["torch", "xformers", "flash", "sageattn"]
attn_mode = globals().get("attn_mode", attn_mode_param)
#@markdown Precisión mixta para el entrenamiento.
precision_param = "bf16" #@param ["full fp16", "full bf16", "mixed fp16", "mixed bf16"]
precision = globals().get("precision", precision_param)
#@markdown Cachear latentes del VAE para liberar VRAM.
cache_latents = True #@param {type:"boolean"}
cache_latents_to_disk = False #@param {type:"boolean"}
#@markdown Cachear salidas del text encoder Qwen3. Recomendado si no se entrena LoRA del text encoder. Desactiva shuffle_tags y el entrenamiento del text encoder.
cache_text_encoder_outputs  = False  # @param {type:"boolean"}

mixed_precision = "no"
if "fp16" in precision:
  mixed_precision = "fp16"
elif "bf16" in precision:
  mixed_precision = "bf16"
full_precision = "full" in precision

#@markdown ### ▶️ Advanced
#@markdown El optimizador utilizado para el entrenamiento.
optimizer_param = "Prodigy" #@param ["AdamW8bit", "Prodigy", "DAdaptation", "DadaptAdam", "DadaptLion", "AdamW", "Lion", "SGDNesterov", "SGDNesterov8bit", "AdaFactor", "Came"]
optimizer = globals().get("optimizer", optimizer_param)
#@markdown Argumentos recomendados para Prodigy: `decouple=True weight_decay=0.01 betas=[0.9,0.999] d_coef=2 use_bias_correction=True safeguard_warmup=True`
#@markdown Si se selecciona Dadapt o Prodigy y se marca la casilla recomendada, se aplicarán valores optimizados.
recommended_values = True #@param {type:"boolean"}
#@markdown Alternativamente, establezca sus propios argumentos de optimizador separados por espacios.
optimizer_args = "" #@param {type:"string"}
optimizer_args = [a.strip() for a in optimizer_args.split(' ') if a]

#@markdown Tipo de pérdida (loss function). `l2` es el valor por defecto para Anima.
loss_type_param = "l2" #@param ["l1", "l2", "huber", "smooth_l1"]
loss_type = globals().get("loss_type", loss_type_param)

if recommended_values:
  if any(opt in optimizer.lower() for opt in ["dadapt", "prodigy"]):
    unet_lr = 1.0
    text_encoder_lr = 1.0
    full_precision = False
    network_alpha = network_dim
  if optimizer == "Prodigy":
    optimizer_args = ["decouple=True", "weight_decay=0.01", "betas=[0.9,0.999]", "d_coef=1", "use_bias_correction=True", "safeguard_warmup=True"]
  elif optimizer == "AdamW8bit":
    optimizer_args = ["weight_decay=0.1", "betas=[0.9,0.99]"]
  elif optimizer == "AdaFactor":
    optimizer_args = ["scale_parameter=False", "relative_step=False", "warmup_init=False"]
  elif optimizer == "Came":
    optimizer_args = ["weight_decay=0.04"]

if optimizer == "Came":
  optimizer = "LoraEasyCustomOptimizer.came.CAME"

lr_scheduler_type = None
lr_scheduler_args = None
lr_scheduler_num_cycles = lr_scheduler_number
lr_scheduler_power = lr_scheduler_number

if "rex" in lr_scheduler:
  lr_scheduler = "cosine"
  lr_scheduler_type = "LoraEasyCustomOptimizer.RexAnnealingWarmRestarts.RexAnnealingWarmRestarts"
  lr_scheduler_args = ["min_lr=1e-9", "gamma=0.9", "d=0.9"]

# Misc
seed = 42
gradient_accumulation_steps = 1
bucket_reso_steps = 64
min_bucket_reso = 256
max_bucket_reso = 4096

#@markdown ### ▶️ Ready
#@markdown Ahora puedes ejecutar esta celda para entrenar tu LoRA de Anima. ¡Buena suerte!

# 👩‍💻 Cool code goes here


for required_dir in (models_dir, downloads_dir):
  os.makedirs(required_dir, exist_ok=True)


def lightning_rel(path):
  try:
    return os.path.relpath(path, root_dir)
  except ValueError:
    return path


venv_python = sys.executable
train_network = os.path.join(kohya_dir, "anima_train_network.py")

if "lora_projects" in folder_structure:
  main_dir      = os.path.join(root_dir, "lora_projects")
  log_folder    = os.path.join(main_dir, "_logs")
  config_folder = os.path.join(main_dir, project_name)
  images_folder = os.path.join(main_dir, project_name, "dataset")
  output_folder = os.path.join(main_dir, project_name, "output")
else:
  main_dir      = os.path.join(root_dir, "lora_training")
  images_folder = os.path.join(main_dir, "datasets", project_name)
  output_folder = os.path.join(main_dir, "output", project_name)
  config_folder = os.path.join(main_dir, "config", project_name)
  log_folder    = os.path.join(main_dir, "log")

config_file = os.path.join(config_folder, "training_config.toml")
dataset_config_file = os.path.join(config_folder, "dataset_config.toml")

def install_trainer():
  global installed
  libtcmalloc_path = os.path.join(root_dir, "libtcmalloc_minimal.so.4")

  if 'installed' not in globals():
    installed = False

  if not os.path.exists(libtcmalloc_path):
    _run_cmd(f"wget -q -c --show-progress https://github.com/camenduru/gperftools/releases/download/v1.0/libtcmalloc_minimal.so.4 -O {libtcmalloc_path}")

  if not os.path.exists(trainer_dir):
    _run_cmd(f"git clone -b dev https://github.com/gwhitez/LoRA_Easy_Training_scripts_Backend.git {trainer_dir}")
  else:
    os.chdir(trainer_dir)
    _run_cmd("git pull")
    os.chdir(root_dir)

  os.chdir(kohya_dir)
  if LOAD_TRUNCATED_IMAGES:
    _run_cmd("sed -i 's/from PIL import Image/from PIL import Image, ImageFile\\nImageFile.LOAD_TRUNCATED_IMAGES=True/g' library/train_util.py")
  if BETTER_EPOCH_NAMES:
    _run_cmd("sed -i 's/{:06d}/{:02d}/g' library/train_util.py")
    train_network_path = Path(kohya_dir) / "train_network.py"
    try:
      text = train_network_path.read_text()
    except FileNotFoundError:
      text = None
    if text is not None:
      pattern = re.compile(r'"-\{:[0-9]+d\}\.".format\(([^)]+)\)\s*\+\s*args\.save_model_as')

      def _repl(match: re.Match) -> str:
        expr = match.group(1).strip()
        return 'f"-{' + expr + ':02d}." + args.save_model_as'

      new_text, count = pattern.subn(_repl, text)
      if count:
        train_network_path.write_text(new_text)
      else:
        print("⚠️ No se pudo parchear train_network.py para los nuevos nombres de checkpoint.")
  if FIX_DIFFUSERS:
    deprecation_utils = os.path.join(kohya_dir, "/home/zeus/miniconda3/envs/cloudspace/lib/python3.12/site-packages/diffusers/utils/deprecation_utils.py")
    _run_cmd(f"sed -i 's/if version.parse/if False:#/g' {deprecation_utils}")
  if FIX_WANDB_WARNING:
    _run_cmd("sed -i 's/accelerator.log(logs, step=epoch + 1)//g' train_network.py")

  os.environ["LD_PRELOAD"] = libtcmalloc_path
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
  os.environ["BITSANDBYTES_NOWELCOME"] = "1"
  os.environ["SAFETENSORS_FAST_GPU"] = "1"
  os.environ["PYTHONWARNINGS"] = "ignore"
  os.chdir(root_dir)

def validate_dataset():
  global lr_warmup_steps, lr_warmup_ratio, caption_extension, keep_tokens, model_url
  supported_types = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

  print("\n💿 Checking dataset...")
  if not project_name.strip() or any(c in project_name for c in " .()\"\'\\/"):
    print("💥 Error: Elija un nombre de proyecto válido.")
    return

  # Find the folders and files
  if custom_dataset:
    try:
      datconf = toml.loads(custom_dataset)
      datasets = [d for d in datconf["datasets"][0]["subsets"]]
    except:
      print(f"💥 Error: El conjunto de datos personalizado no es válido o contiene un error. Por favor, compruebe la plantilla original.")
      return
    reg = [d.get("image_dir") for d in datasets if d.get("is_reg", False)]
    datasets_dict = {d["image_dir"]: d["num_repeats"] for d in datasets}
    folders = datasets_dict.keys()
    files = [f for folder in folders for f in os.listdir(folder)]
    images_repeats = {folder: (len([f for f in os.listdir(folder) if f.lower().endswith(supported_types)]), datasets_dict[folder]) for folder in folders}
  else:
    reg = []
    folders = [images_folder]
    files = os.listdir(images_folder)
    images_repeats = {images_folder: (len([f for f in files if f.lower().endswith(supported_types)]), num_repeats)}

  # Validation
  for folder in folders:
    if not os.path.exists(folder):
      print(f"💥 Error: La carpeta {lightning_rel(folder)} no existe.")
      return
  for folder, (img, rep) in images_repeats.items():
    if not img:
      print(f"💥 Error: tú {lightning_rel(folder)} La carpeta está vacía.")
      return
  test_files = []
  for f in files:
    if not f.lower().endswith((caption_extension, ".npz")) and not f.lower().endswith(supported_types):
      print(f"💥 Error: Archivo no válido en el conjunto de datos: \"{f}\". Abortar.")
      return
    for ff in test_files:
      if f.endswith(supported_types) and ff.endswith(supported_types) \
          and os.path.splitext(f)[0] == os.path.splitext(ff)[0]:
        print(f"💥 Error: Los archivos {f} y {ff} no puede tener el mismo nombre. Abortar.")
        return
    test_files.append(f)

  if caption_extension and not [txt for txt in files if txt.lower().endswith(caption_extension)]:
    caption_extension = ""

  # Show estimations to the user

  pre_steps_per_epoch = sum(img*rep for (img, rep) in images_repeats.values())
  steps_per_epoch = pre_steps_per_epoch/train_batch_size
  total_steps = max_train_steps or int(max_train_epochs*steps_per_epoch)
  estimated_epochs = int(total_steps/steps_per_epoch)
  lr_warmup_steps = int(total_steps*lr_warmup_ratio)

  for folder, (img, rep) in images_repeats.items():
    print("📁" + lightning_rel(folder) + (" (Regularization)" if folder in reg else ""))
    print(f"📈 Se encontró {img} imágenes con {rep} repeticiones, igual {img*rep} pasos.")
  print(f"📉 Divide {pre_steps_per_epoch} pasos por {train_batch_size} batch size para obtener {steps_per_epoch} pasos por epoch.")
  if max_train_epochs:
    print(f"🔮 Habrá {max_train_epochs} epochs, por alrededor de {total_steps} total de pasos.")
  else:
    print(f"🔮 Habrá {total_steps} pasos, divididos en {estimated_epochs} epochs y algo más.")

  if total_steps > 10000:
    print("💥 Error: El total de pasos es demasiado alto. Probablemente cometiste un error. Abortar...")
    return

  return True

def create_config():
  global dataset_config_file, config_file, model_file

  # Build network_args list for Anima
  _network_args = []
  if train_llm_adapter:
    _network_args.append("train_llm_adapter=True")
  if exclude_patterns:
    _network_args.append(f"exclude_patterns=['{exclude_patterns}']")
  if include_patterns:
    _network_args.append(f"include_patterns=['{include_patterns}']")
  if network_reg_dims:
    _network_args.append(f"network_reg_dims={network_reg_dims}")
  if network_reg_lrs:
    _network_args.append(f"network_reg_lrs={network_reg_lrs}")

  if override_config_file:
    config_file = override_config_file
    print(f"\n⭕ Using custom config file {config_file}")
  else:
    config_dict = {
      "network_arguments": {
        "unet_lr": unet_lr,
        "text_encoder_lr": text_encoder_lr if not cache_text_encoder_outputs else 0,
        "network_dim": network_dim,
        "network_alpha": network_alpha,
        "network_module": network_module,
        "network_args": _network_args if _network_args else None,
        "network_train_unet_only": network_train_unet_only or cache_text_encoder_outputs,
      },
      "optimizer_arguments": {
        "learning_rate": unet_lr,
        "lr_scheduler": lr_scheduler,
        "lr_scheduler_type": lr_scheduler_type,
        "lr_scheduler_args": lr_scheduler_args,
        "lr_scheduler_num_cycles": lr_scheduler_num_cycles if lr_scheduler == "cosine_with_restarts" else None,
        "lr_scheduler_power": lr_scheduler_power if lr_scheduler == "polynomial" else None,
        "lr_warmup_steps": lr_warmup_steps if lr_scheduler not in ("cosine", "constant") else None,
        "optimizer_type": optimizer,
        "optimizer_args": optimizer_args or None,
        "loss_type": loss_type,
        "max_grad_norm": 1.0,
      },
      "training_arguments": {
        "lowram": LOWRAM,
        "pretrained_model_name_or_path": model_file,
        "qwen3": qwen3_path if qwen3_path else None,
        "vae": vae_file if vae_file else None,
        "llm_adapter_path": llm_adapter_path if llm_adapter_path else None,
        "t5_tokenizer_path": t5_tokenizer_path if t5_tokenizer_path else None,
        "max_train_steps": max_train_steps,
        "max_train_epochs": max_train_epochs,
        "train_batch_size": train_batch_size,
        "seed": seed,
        "timestep_sampling": timestep_sampling,
        "discrete_flow_shift": discrete_flow_shift,
        "sigmoid_scale": sigmoid_scale,
        "weighting_scheme": weighting_scheme,
        "qwen3_max_token_length": qwen3_max_token_length,
        "t5_max_token_length": t5_max_token_length,
        "attn_mode": attn_mode if attn_mode != "torch" else None,
        "split_attn": True if attn_mode == "xformers" else None,
        "ip_noise_gamma": ip_noise_gamma if ip_noise_gamma_enabled else None,
        "gradient_checkpointing": True,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "max_data_loader_n_workers": 1,
        "persistent_data_loader_workers": True,
        "mixed_precision": mixed_precision,
        "full_fp16": mixed_precision == "fp16" and full_precision,
        "full_bf16": mixed_precision == "bf16" and full_precision,
        "cache_latents": cache_latents,
        "cache_latents_to_disk": cache_latents_to_disk,
        "cache_text_encoder_outputs": cache_text_encoder_outputs,
        "blocks_to_swap": blocks_to_swap if blocks_to_swap > 0 else None,
        "vae_chunk_size": vae_chunk_size if vae_chunk_size > 0 else None,
        "vae_disable_cache": vae_disable_cache or None,
        "unsloth_offload_checkpointing": unsloth_offload_checkpointing or None,
      },
      "saving_arguments": {
        "save_precision": "fp16",
        "save_model_as": "safetensors",
        "save_every_n_epochs": save_every_n_epochs,
        "save_last_n_epochs": keep_only_last_n_epochs,
        "output_name": project_name,
        "output_dir": output_folder,
        "log_prefix": project_name,
        "logging_dir": log_folder,
        "wandb_api_key": wandb_key or None,
        "log_with": "wandb" if wandb_key else None,
      }
    }

    for key in config_dict:
      if isinstance(config_dict[key], dict):
        config_dict[key] = {k: v for k, v in config_dict[key].items() if v is not None}

    with open(config_file, "w") as f:
      f.write(toml.dumps(config_dict))
    print(f"\n📄 Config saved to {config_file}")

  if override_dataset_config_file:
    dataset_config_file = override_dataset_config_file
    print(f"⭕ Using custom dataset config file {dataset_config_file}")
  else:
    dataset_config_dict = {
      "general": {
        "resolution": resolution,
        "shuffle_caption": shuffle_caption and not cache_text_encoder_outputs,
        "keep_tokens": keep_tokens,
        "flip_aug": False,
        "caption_extension": caption_extension,
        "enable_bucket": True,
        "bucket_no_upscale": False,
        "bucket_reso_steps": bucket_reso_steps,
        "min_bucket_reso": min_bucket_reso,
        "max_bucket_reso": max_bucket_reso,
      },
      "datasets": toml.loads(custom_dataset)["datasets"] if custom_dataset else [
        {
          "subsets": [
            {
              "num_repeats": num_repeats,
              "image_dir": images_folder,
              "class_tokens": None if caption_extension else project_name
            }
          ]
        }
      ]
    }

    for key in dataset_config_dict:
      if isinstance(dataset_config_dict[key], dict):
        dataset_config_dict[key] = {k: v for k, v in dataset_config_dict[key].items() if v is not None}

    with open(dataset_config_file, "w") as f:
      f.write(toml.dumps(dataset_config_dict))
    print(f"📄 Configuración de dataset guardada en {dataset_config_file}")

def download_anima_components() -> bool:
  """Download Qwen3 text encoder and Qwen-Image VAE if not already present."""
  os.makedirs(models_dir, exist_ok=True)

  # Download Qwen3 text encoder
  if not os.path.exists(qwen3_path):
    print(f"🌐 Descargando Qwen3-0.6B text encoder en {qwen3_path} ...")
    try:
      _run_cmd(f"aria2c '{qwen3_url}' --console-log-level=warn -c -s 16 -x 16 -k 10M -d {models_dir} -o '{os.path.basename(qwen3_path)}'")
    except Exception as exc:
      print(f"💥 Error al descargar Qwen3 text encoder: {exc}")
      return False
    if not os.path.exists(qwen3_path):
      print(f"💥 Error: Qwen3 text encoder no se encontró después de la descarga: {qwen3_path}")
      return False
  print(f"✅ Qwen3 text encoder listo: {qwen3_path}")

  # Download Qwen-Image VAE
  if not os.path.exists(anima_vae_path):
    print(f"🌐 Descargando Qwen-Image VAE en {anima_vae_path} ...")
    try:
      _run_cmd(f"aria2c '{anima_vae_url}' --console-log-level=warn -c -s 16 -x 16 -k 10M -d {models_dir} -o '{os.path.basename(anima_vae_path)}'")
    except Exception as exc:
      print(f"💥 Error al descargar Qwen-Image VAE: {exc}")
      return False
    if not os.path.exists(anima_vae_path):
      print(f"💥 Error: Qwen-Image VAE no se encontró después de la descarga: {anima_vae_path}")
      return False
  print(f"✅ Qwen-Image VAE listo: {anima_vae_path}")

  if llm_adapter_path:
    if os.path.exists(llm_adapter_path):
      print(f"✅ LLM Adapter listo: {llm_adapter_path}")
    else:
      print(f"⚠️ LLM Adapter especificado pero no encontrado: {llm_adapter_path}")
  return True

def download_model():
  global old_model_url, model_url, model_file

  real_model_url = (model_url or "").strip()
  if not real_model_url:
    print("💥 Error: no se especificó ningún modelo base para entrenar.")
    return False

  # Check local path first
  local_candidate = None
  if '://' not in real_model_url:
    candidate = Path(real_model_url)
    if not candidate.is_absolute():
      candidate = Path(root_dir) / real_model_url.lstrip('/')
    if candidate.exists():
      local_candidate = candidate
    else:
      print(f"💥 Error: el modelo local {candidate} no existe. Asegúrate de que esté dentro de {root_dir} o usa una URL.")
      return False

  if local_candidate is not None:
    model_file = str(local_candidate)
    print(f"📁 Usando modelo local: {model_file}")
  else:
    if real_model_url.lower().endswith((".ckpt", ".safetensors")):
      filename = os.path.basename(real_model_url)
    else:
      filename = "downloaded_model.safetensors"

    model_file = os.path.join(models_dir, filename)
    if os.path.exists(model_file):
      _run_cmd(f"rm '{model_file}'")

    if re.search(r"(?:https?://)?(?:www\.)?huggingface\.co/[^/]+/[^/]+/blob", real_model_url):
      real_model_url = real_model_url.replace("blob", "resolve")

    print(f"🌐 Descargando modelo DiT en {model_file} ...")
    _run_cmd(f"aria2c '{real_model_url}' --console-log-level=warn -c -s 16 -x 16 -k 10M -d {models_dir} -o '{os.path.basename(model_file)}'")

  if model_file.lower().endswith(".safetensors"):
    from safetensors.torch import load_file as load_safetensors
    try:
      test = load_safetensors(model_file)
      del test
    except Exception as exc:
      print(f"💥 Error al validar el archivo safetensors {model_file}: {exc}")
      return False

  return True


def calculate_rex_steps():
  # https://github.com/derrian-distro/LoRA_Easy_Training_scripts_Backend/blob/c34084b0435e6e19bb7a01ac1ecbadd185ee8c1e/utils/validation.py#L268
  global max_train_steps
  print("\n🤔 Calculating Rex steps")
  if max_train_steps:
    calculated_max_steps = max_train_steps
  else:
    from library.train_util import BucketManager
    from PIL import Image
    from pathlib import Path
    import math

    with open(dataset_config_file, "r") as f:
      subsets = toml.load(f)["datasets"][0]["subsets"]

    supported_types = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]
    res = (resolution, resolution)
    bucketManager = BucketManager(False, res, min_bucket_reso, max_bucket_reso, bucket_reso_steps)
    bucketManager.make_buckets()
    for subset in subsets:
        for image in Path(subset["image_dir"]).iterdir():
            if image.suffix not in supported_types:
                continue
            with Image.open(image) as img:
                bucket_reso, _, _ = bucketManager.select_bucket(img.width, img.height)
                for _ in range(subset["num_repeats"]):
                    bucketManager.add_image(bucket_reso, image)
    steps_before_acc = sum(math.ceil(len(bucket) / train_batch_size) for bucket in bucketManager.buckets)
    calculated_max_steps = math.ceil(steps_before_acc / gradient_accumulation_steps) * max_train_epochs
    del bucketManager

  cycle_steps = calculated_max_steps // (lr_scheduler_num_cycles or 1)
  print(f"  cycle steps: {cycle_steps}")
  lr_scheduler_args.append(f"first_cycle_max_steps={cycle_steps}")

  warmup_steps = round(calculated_max_steps * lr_warmup_ratio) // (lr_scheduler_num_cycles or 1)
  if warmup_steps > 0:
    print(f"  warmup steps: {warmup_steps}")
    lr_scheduler_args.append(f"warmup_steps={warmup_steps}")

def main():
  global dependencies_installed

  for dir in (main_dir, trainer_dir, log_folder, images_folder, output_folder, config_folder, models_dir, downloads_dir):
    os.makedirs(dir, exist_ok=True)

  if not validate_dataset():
    return

  if not dependencies_installed:
    print("🏭 Instalando entrenador...")
    t0 = time.time()
    install_trainer()
    t1 = time.time()
    dependencies_installed = True
    print(f"✅ Instalación terminada en {int(t1 - t0)} segundos.")
  else:
    print("✅ Dependencias ya instaladas.")

  if old_model_url != model_url or not model_file or not os.path.exists(model_file):
    print("🔄 Obteniendo modelo DiT...")
    if not download_model():
      print("💥 Error: el modelo que especificó no es válido o está corrupto. Verifique que la URL sea accesible o que la ruta exista dentro de su espacio Lightning.")
      return
    print()
  else:
    print("🔄 Modelo DiT ya disponible.")

  print("🔄 Verificando componentes Anima (Qwen3 text encoder + Qwen-Image VAE)...")
  if not download_anima_components():
    print("💥 Error: no se pudieron descargar los componentes de Anima.")
    return

  if lr_scheduler_type:
    create_config()
    os.chdir(kohya_dir)
    calculate_rex_steps()
    os.chdir(root_dir)

  create_config()

  print("⭐ Iniciando Entrenador Anima LoRA..")

  os.chdir(kohya_dir)
  _run_cmd(f"{venv_python} {train_network} --console_log_simple --config_file={config_file} --dataset_config={dataset_config_file}")
  os.chdir(root_dir)

  if not get_ipython().__dict__.get('user_ns', {}).get('_exit_code', False):
    display(Markdown(f"### ✅ ¡Hecho! Tus archivos se encuentran en `{output_folder}`"))

main()
print("🔵 El cuaderno continuará en ejecución. Detén manualmente la sesión de Lightning cuando termines.")
