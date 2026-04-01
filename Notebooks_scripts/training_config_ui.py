"""UI helpers for configuring the Anima LoRA training session from the notebook.

Adapted from the original SDXL training UI to support Anima (DiT-based
architecture with Qwen3 text encoder and Qwen-Image VAE).  The only requirement
from the notebook is to call :func:`render_quick_training_config` passing
``globals()`` so the widgets can read and update the shared state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple

import os

import ipywidgets as widgets
from IPython.display import Markdown, display
from pathlib import Path


PROJECTS_ROOT = Path("/teamspace/studios/this_studio/lora_projects")
MODELS_ROOT = Path("/teamspace/studios/this_studio/models")


@dataclass(frozen=True)
class _DropdownOption:
    label: str
    value: Any

    def as_tuple(self) -> Tuple[str, Any]:
        return self.label, self.value


def _format_scientific(value: Any) -> str:
    """Return ``value`` formatted in scientific notation when possible."""
    try:
        return format(float(value), ".0e")
    except (TypeError, ValueError):
        return str(value)


def _resolve_precision(default: str, options: Iterable[str]) -> str:
    if default in options:
        return default
    default = str(default)
    if "bf16" in default:
        return "mixed bf16"
    if "fp16" in default:
        return "mixed fp16"
    return next(iter(options))


def render_quick_training_config(namespace: Dict[str, Any]) -> None:
    """Render the interactive configuration UI for Anima LoRA training.

    Parameters
    ----------
    namespace:
        Usually the ``globals()`` dictionary from the notebook.  The widgets
        will read default values from it and write the updated selections back
        into the same mapping so that the following cells behave exactly like
        the original implementation.
    """


    lr_scheduler_options = [
        "constant",
        "cosine",
        "cosine_with_restarts",
        "constant_with_warmup",
        "rex",
    ]

    precision_options = [
        "full fp16",
        "full bf16",
        "mixed fp16",
        "mixed bf16",
    ]

    optimizer_options = [
        "AdamW8bit",
        "Prodigy",
        "DAdaptation",
        "DadaptAdam",
        "DadaptLion",
        "AdamW",
        "AdaFactor",
        "Came",
    ]

    timestep_sampling_options = [
        "sigma",
        "uniform",
        "sigmoid",
        "shift",
        "flux_shift",
        "logit_normal",
    ]

    attn_mode_options = [
        "torch",
        "xformers",
        "flash",
        "sageattn",
    ]

    weighting_scheme_options = [
        "uniform",
        "sigma_sqrt",
        "cosmap",
        "none",
        "logit_normal",
    ]

    loss_type_options = [
        "l1",
        "l2",
        "huber",
        "smooth_l1",
    ]

    # Widgets -----------------------------------------------------------------
    base_style = {"description_width": "180px"}
    number_layout = widgets.Layout(width="100%")

    project_name_widget = widgets.Text(
        value=str(namespace.get("project_name", "")),
        description="project_name",
        placeholder="Nombre del proyecto",
        style=base_style,
    )

    # --- Anima-specific component paths (only optional ones, qwen3/vae auto-download) ---
    llm_adapter_path_widget = widgets.Text(
        value=str(namespace.get("llm_adapter_path", "")),
        description="llm_adapter_path",
        placeholder="Opcional: ruta al LLM Adapter",
        style=base_style,
        layout=widgets.Layout(width="100%"),
    )
    t5_tokenizer_path_widget = widgets.Text(
        value=str(namespace.get("t5_tokenizer_path", "")),
        description="t5_tokenizer_path",
        placeholder="Opcional: ruta al T5 tokenizer",
        style=base_style,
        layout=widgets.Layout(width="100%"),
    )

    resolution_widget = widgets.IntText(
        value=int(namespace.get("resolution", 1024)),
        description="resolution",
        style=base_style,
        layout=number_layout,
    )
    num_repeats_widget = widgets.IntText(
        value=int(namespace.get("num_repeats", 2)),
        description="num_repeats",
        style=base_style,
        layout=number_layout,
    )
    how_many_widget = widgets.IntText(
        value=int(namespace.get("how_many", 40)),
        description="how_many",
        style=base_style,
        layout=number_layout,
    )
    unet_lr_widget = widgets.Text(
        value=_format_scientific(namespace.get("unet_lr", 1e-4)),
        description="unet_lr (DiT)",
        style=base_style,
    )
    text_encoder_lr_widget = widgets.Text(
        value=_format_scientific(namespace.get("text_encoder_lr", 5e-5)),
        description="text_encoder_lr (Qwen3)",
        style=base_style,
    )
    lr_scheduler_widget = widgets.Dropdown(
        options=lr_scheduler_options,
        value=namespace.get("lr_scheduler", "cosine"),
        description="lr_scheduler",
        style=base_style,
    )
    network_dim_widget = widgets.IntText(
        value=int(namespace.get("network_dim", 8)),
        description="network_dim",
        style=base_style,
        layout=number_layout,
    )
    network_alpha_widget = widgets.IntText(
        value=int(namespace.get("network_alpha", 4)),
        description="network_alpha",
        style=base_style,
        layout=number_layout,
    )
    train_batch_size_widget = widgets.IntText(
        value=int(namespace.get("train_batch_size", 4)),
        description="train_batch_size",
        style=base_style,
        layout=number_layout,
    )
    precision_widget = widgets.Dropdown(
        options=precision_options,
        value=_resolve_precision(namespace.get("precision", "bf16"), precision_options),
        description="precision",
        style=base_style,
    )
    optimizer_widget = widgets.Dropdown(
        options=optimizer_options,
        value=namespace.get("optimizer", "Prodigy"),
        description="optimizer",
        style=base_style,
    )

    # --- Anima-specific training parameters ---
    timestep_sampling_widget = widgets.Dropdown(
        options=timestep_sampling_options,
        value=namespace.get("timestep_sampling", "sigmoid"),
        description="timestep_sampling",
        style=base_style,
    )
    discrete_flow_shift_widget = widgets.FloatText(
        value=float(namespace.get("discrete_flow_shift", 3.0)),
        description="discrete_flow_shift",
        style=base_style,
        layout=number_layout,
    )
    sigmoid_scale_widget = widgets.FloatText(
        value=float(namespace.get("sigmoid_scale", 1.0)),
        description="sigmoid_scale",
        style=base_style,
        layout=number_layout,
    )
    weighting_scheme_widget = widgets.Dropdown(
        options=weighting_scheme_options,
        value=namespace.get("weighting_scheme", "uniform"),
        description="weighting_scheme",
        style=base_style,
    )
    loss_type_widget = widgets.Dropdown(
        options=loss_type_options,
        value=namespace.get("loss_type", "l2"),
        description="loss_type",
        style=base_style,
    )
    attn_mode_widget = widgets.Dropdown(
        options=attn_mode_options,
        value=namespace.get("attn_mode", "torch"),
        description="attn_mode",
        style=base_style,
    )

    # --- Memory optimization ---
    blocks_to_swap_widget = widgets.IntText(
        value=int(namespace.get("blocks_to_swap", 0)),
        description="blocks_to_swap",
        style=base_style,
        layout=number_layout,
    )
    vae_chunk_size_widget = widgets.IntText(
        value=int(namespace.get("vae_chunk_size", 64)),
        description="vae_chunk_size",
        style=base_style,
        layout=number_layout,
    )
    vae_disable_cache_widget = widgets.Checkbox(
        value=bool(namespace.get("vae_disable_cache", True)),
        description="vae_disable_cache",
        style={"description_width": "180px"},
        indent=False,
    )
    unsloth_offload_widget = widgets.Checkbox(
        value=bool(namespace.get("unsloth_offload_checkpointing", False)),
        description="unsloth_offload_checkpointing",
        style={"description_width": "180px"},
        indent=False,
    )

    # --- network_train_unet_only ---
    network_train_unet_only_widget = widgets.Checkbox(
        value=bool(namespace.get("network_train_unet_only", True)),
        description="network_train_unet_only",
        style={"description_width": "180px"},
        indent=False,
    )

    # --- LLM Adapter & Module control ---
    train_llm_adapter_widget = widgets.Checkbox(
        value=bool(namespace.get("train_llm_adapter", False)),
        description="train_llm_adapter",
        style={"description_width": "180px"},
        indent=False,
    )
    qwen3_max_token_length_widget = widgets.IntText(
        value=int(namespace.get("qwen3_max_token_length", 512)),
        description="qwen3_max_token_len",
        style=base_style,
        layout=number_layout,
    )
    t5_max_token_length_widget = widgets.IntText(
        value=int(namespace.get("t5_max_token_length", 512)),
        description="t5_max_token_len",
        style=base_style,
        layout=number_layout,
    )

    status_output = widgets.HTML()
    apply_button = widgets.Button(
        description="Aplicar parámetros",
        button_style="success",
        icon="check",
        layout=widgets.Layout(width="auto", align_self="flex-end"),
    )

    grid_layout = widgets.Layout(
        grid_template_columns="repeat(2, minmax(0, 1fr))",
        grid_gap="12px",
        width="100%",
    )
    basics_grid = widgets.GridBox(
        children=[
            project_name_widget,
            resolution_widget,
            num_repeats_widget,
            how_many_widget,
        ],
        layout=grid_layout,
    )

    anima_components_grid = widgets.GridBox(
        children=[
            llm_adapter_path_widget,
            t5_tokenizer_path_widget,
        ],
        layout=grid_layout,
    )

    advanced_grid = widgets.GridBox(
        children=[
            unet_lr_widget,
            text_encoder_lr_widget,
            lr_scheduler_widget,
            network_dim_widget,
            network_alpha_widget,
            train_batch_size_widget,
            precision_widget,
            optimizer_widget,
        ],
        layout=grid_layout,
    )

    anima_params_grid = widgets.GridBox(
        children=[
            timestep_sampling_widget,
            discrete_flow_shift_widget,
            sigmoid_scale_widget,
            weighting_scheme_widget,
            loss_type_widget,
            attn_mode_widget,
            qwen3_max_token_length_widget,
            t5_max_token_length_widget,
        ],
        layout=grid_layout,
    )

    memory_grid = widgets.GridBox(
        children=[
            blocks_to_swap_widget,
            vae_chunk_size_widget,
            vae_disable_cache_widget,
            unsloth_offload_widget,
            train_llm_adapter_widget,
            network_train_unet_only_widget,
        ],
        layout=grid_layout,
    )


    def apply_params(_=None) -> None:
        try:
            updates = {
                "project_name": project_name_widget.value.strip(),
                "llm_adapter_path": llm_adapter_path_widget.value.strip(),
                "t5_tokenizer_path": t5_tokenizer_path_widget.value.strip(),
                "resolution": int(resolution_widget.value),
                "num_repeats": int(num_repeats_widget.value),
                "how_many": int(how_many_widget.value),
                "unet_lr": float(unet_lr_widget.value),
                "text_encoder_lr": float(text_encoder_lr_widget.value),
                "lr_scheduler": lr_scheduler_widget.value,
                "network_dim": int(network_dim_widget.value),
                "network_alpha": int(network_alpha_widget.value),
                "train_batch_size": int(train_batch_size_widget.value),
                "precision": precision_widget.value,
                "optimizer": optimizer_widget.value,
                "timestep_sampling": timestep_sampling_widget.value,
                "discrete_flow_shift": float(discrete_flow_shift_widget.value),
                "sigmoid_scale": float(sigmoid_scale_widget.value),
                "weighting_scheme": weighting_scheme_widget.value,
                "loss_type": loss_type_widget.value,
                "attn_mode": attn_mode_widget.value,
                "blocks_to_swap": int(blocks_to_swap_widget.value),
                "vae_chunk_size": int(vae_chunk_size_widget.value),
                "vae_disable_cache": bool(vae_disable_cache_widget.value),
                "unsloth_offload_checkpointing": bool(unsloth_offload_widget.value),
                "train_llm_adapter": bool(train_llm_adapter_widget.value),
                "network_train_unet_only": bool(network_train_unet_only_widget.value),
                "qwen3_max_token_length": int(qwen3_max_token_length_widget.value),
                "t5_max_token_length": int(t5_max_token_length_widget.value),
            }
        except ValueError as exc:
            status_output.value = f"<b>Error:</b> {exc}"
            return

        namespace.update(updates)
        unet_lr_widget.value = _format_scientific(updates["unet_lr"])
        text_encoder_lr_widget.value = _format_scientific(updates["text_encoder_lr"])

        project_name_value = updates["project_name"]
        directory_message = ""
        if project_name_value:
            project_dir = PROJECTS_ROOT / project_name_value
            dataset_dir = project_dir / "dataset"
            try:
                dataset_dir.mkdir(parents=True, exist_ok=True)
                directory_message = f" Directorios preparados en <code>{project_dir}</code>."
            except Exception as exc:  # pragma: no cover - user environment errors
                status_output.value = f"<b>Error al preparar directorios:</b> {exc}"
                return

        status_output.value = f"<b>Parámetros actualizados correctamente.</b>{directory_message}"

    apply_button.on_click(apply_params)

    display(Markdown("""
### Configuración Anima LoRA Training
Personaliza tu sesión de entrenamiento Anima desde este panel. El modelo DiT, Qwen3 y VAE se descargan automáticamente.
Haz clic en **Aplicar parámetros** para guardar los cambios.
"""))
    display(
        widgets.VBox(
            [
                widgets.HTML("<h4 style='margin-bottom:4px;'>Datos básicos</h4>"),
                basics_grid,
                widgets.HTML("<h4 style='margin:16px 0 4px;'>Componentes opcionales</h4>"),
                anima_components_grid,
                widgets.HTML("<h4 style='margin:16px 0 4px;'>Ajustes de Entrenamiento</h4>"),
                advanced_grid,
                widgets.HTML("<h4 style='margin:16px 0 4px;'>Parámetros Anima</h4>"),
                anima_params_grid,
                widgets.HTML("<h4 style='margin:16px 0 4px;'>Memoria y Optimización</h4>"),
                memory_grid,
                widgets.HBox([widgets.HBox([], layout=widgets.Layout(flex="1")), apply_button]),
                status_output,
            ]
        )
    )

    apply_params()


__all__ = ["render_quick_training_config"]
