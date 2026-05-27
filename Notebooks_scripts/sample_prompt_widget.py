"""
Sample Prompt Generator — ipywidgets UI for Jupyter Notebooks.

Usage (in a notebook cell):
    %run /root/sample_prompt/sample_prompt_widget.py

Or import:
    from sample_prompt_widget import create_sample_prompt_ui
    display(create_sample_prompt_ui())
"""

import os
import ipywidgets as widgets
from IPython.display import display, HTML


# ── Default values ────────────────────────────────────────────────────────────
_DEFAULTS = {
    "positive_prompt": (
        "n1a, masterpiece, score_8, 1girl, grey hair, facial mark, "
        "yellow jumpsuit, loose belt, dagger, sheathed, white gloves, "
        "hand on own hip, teardrop gem, dangling jewelry, red gem"
    ),
    "negative_prompt": "low quality, worst quality",
    "resolution": "768x1344",
    "seed": 5000,
    "cfg_scale": 4.5,
    "steps": 23,
    "output_dir": "/root/samplefile",
}

_RESOLUTION_OPTIONS = [
    "832x1216",
    "1216x832",
    "768x1344",
    "1344x768",
    "768x1024",
    "1024x768",
    "1024x1024",
    "896x1152",
    "1152x896",
]


# ── Styles ────────────────────────────────────────────────────────────────────
_TEXTAREA_LAYOUT = widgets.Layout(width="100%", min_height="80px")
_INPUT_LAYOUT = widgets.Layout(width="100%")
_BUTTON_LAYOUT = widgets.Layout(width="220px", height="38px")
_BOX_LAYOUT = widgets.Layout(
    width="100%",
    padding="16px",
    border="1px solid #444",
    border_radius="10px",
)

_CSS = """
<style>
.sample-prompt-ui .widget-label { font-weight: 600; color: #ccc; }
.sample-prompt-ui .widget-textarea textarea,
.sample-prompt-ui .widget-text input,
.sample-prompt-ui .widget-dropdown select {
    background: #1e1e2e !important;
    color: #e0e0e0 !important;
    border: 1px solid #555 !important;
    border-radius: 6px !important;
    padding: 6px 10px !important;
}
.sample-prompt-ui .widget-button button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    cursor: pointer !important;
    transition: opacity 0.2s !important;
}
.sample-prompt-ui .widget-button button:hover {
    opacity: 0.85 !important;
}
</style>
"""


# ── UI builder ────────────────────────────────────────────────────────────────
def create_sample_prompt_ui(defaults: dict | None = None) -> widgets.VBox:
    """Build and return the sample-prompt configuration widget.

    Parameters
    ----------
    defaults : dict, optional
        Override any key in ``_DEFAULTS`` (e.g. ``{"seed": 42}``).

    Returns
    -------
    widgets.VBox
        The assembled widget — call ``display(...)`` to render it.
    """
    cfg = {**_DEFAULTS, **(defaults or {})}

    # ── Widgets ───────────────────────────────────────────────────────────
    w_positive = widgets.Textarea(
        value=cfg["positive_prompt"],
        description="Prompt (+):",
        style={"description_width": "initial"},
        layout=_TEXTAREA_LAYOUT,
        placeholder="Prompt positivo para las imágenes de muestra…",
    )

    w_negative = widgets.Textarea(
        value=cfg["negative_prompt"],
        description="Prompt (−):",
        style={"description_width": "initial"},
        layout=_TEXTAREA_LAYOUT,
        placeholder="Prompt negativo para las imágenes de muestra…",
    )

    w_resolution = widgets.Dropdown(
        options=_RESOLUTION_OPTIONS,
        value=cfg["resolution"],
        description="Resolución:",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="320px"),
    )

    w_seed = widgets.IntText(
        value=int(cfg["seed"]),
        description="Seed:",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="220px"),
    )

    w_cfg_scale = widgets.FloatText(
        value=float(cfg["cfg_scale"]),
        description="CFG Scale:",
        step=0.5,
        style={"description_width": "initial"},
        layout=widgets.Layout(width="220px"),
    )

    w_steps = widgets.IntText(
        value=int(cfg["steps"]),
        description="Steps:",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="220px"),
    )

    w_output_dir = widgets.Text(
        value=cfg["output_dir"],
        description="Output dir:",
        style={"description_width": "initial"},
        layout=_INPUT_LAYOUT,
        placeholder="Directorio donde se guardará sample_prompts.txt",
    )

    w_button = widgets.Button(
        description="💾  Guardar Prompt",
        tooltip="Genera el archivo sample_prompts.txt con los parámetros configurados",
        layout=_BUTTON_LAYOUT,
        button_style="",
    )

    w_output = widgets.Output(layout=widgets.Layout(width="100%"))

    # ── Callback ──────────────────────────────────────────────────────────
    def _on_save(_btn):
        w_output.clear_output()
        with w_output:
            try:
                out_dir = w_output_dir.value.strip()
                os.makedirs(out_dir, exist_ok=True)
                filepath = os.path.join(out_dir, "sample_prompts.txt")

                res_w, res_h = w_resolution.value.split("x")
                prompt_line = (
                    f"{w_positive.value.strip()} "
                    f"--n {w_negative.value.strip()} "
                    f"--w {res_w} "
                    f"--h {res_h} "
                    f"--d {w_seed.value} "
                    f"--l {w_cfg_scale.value} "
                    f"--s {w_steps.value}"
                )

                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(prompt_line + "\n")

                display(
                    HTML(
                        f'<div style="background:#1a3a1a; color:#4ade80; '
                        f'padding:10px 14px; border-radius:8px; margin-top:8px; '
                        f'font-family:monospace; font-size:13px;">'
                        f"✅ <b>Guardado correctamente</b><br>"
                        f"📂 <code>{filepath}</code><br><br>"
                        f"<b>Contenido:</b><br>"
                        f"<code>{prompt_line}</code></div>"
                    )
                )
            except Exception as exc:
                display(
                    HTML(
                        f'<div style="background:#3a1a1a; color:#f87171; '
                        f'padding:10px 14px; border-radius:8px; margin-top:8px; '
                        f'font-family:monospace; font-size:13px;">'
                        f"❌ <b>Error:</b> {exc}</div>"
                    )
                )

    w_button.on_click(_on_save)

    # ── Layout ────────────────────────────────────────────────────────────
    header = widgets.HTML(
        value=(
            '<h3 style="margin:0 0 4px 0; color:#a78bfa;">'
            "🎨 Sample Prompt Generator</h3>"
            '<p style="margin:0 0 12px 0; color:#888; font-size:13px;">'
            "Configura y guarda el archivo <code>sample_prompts.txt</code> "
            "para la generación de muestras durante el entrenamiento.</p>"
        )
    )

    numeric_row = widgets.HBox(
        [w_seed, w_cfg_scale, w_steps],
        layout=widgets.Layout(gap="12px"),
    )

    resolution_row = widgets.HBox(
        [w_resolution],
        layout=widgets.Layout(gap="12px"),
    )

    container = widgets.VBox(
        [
            widgets.HTML(_CSS),
            header,
            w_positive,
            w_negative,
            widgets.HTML('<hr style="border-color:#333; margin:8px 0;">'),
            resolution_row,
            numeric_row,
            widgets.HTML('<hr style="border-color:#333; margin:8px 0;">'),
            w_output_dir,
            widgets.HBox(
                [w_button],
                layout=widgets.Layout(justify_content="flex-start", margin="8px 0 0 0"),
            ),
            w_output,
        ],
        layout=_BOX_LAYOUT,
    )
    container.add_class("sample-prompt-ui")

    return container


# ── Auto-display when executed with %run ──────────────────────────────────────
if __name__ == "__main__" or "__file__" not in dir():
    ui = create_sample_prompt_ui()
    display(ui)
