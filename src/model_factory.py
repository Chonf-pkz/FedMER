from typing import Dict, Tuple

MODEL_CHOICES: Tuple[str, ...] = (
    "fedalmer",
    "memocmt",
    "fleser",
    "threemser",
    "cemobam",
)

MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "fedalmer": "FedalMER",
    "memocmt": "MemoCMT",
    "fleser": "FleSER",
    "threemser": "ThreeMSER",
    "cemobam": "CemoBAM",
}

MODEL_ALIASES: Dict[str, str] = {
    "fedalmer": "fedalmer",
    "fedalmermodel": "fedalmer",
    "memocmt": "memocmt",
    "memo_cmt": "memocmt",
    "fleser": "fleser",
    "flexiblemmser": "fleser",
    "threemser": "threemser",
    "three_mser": "threemser",
    "cemobam": "cemobam",
    "multimodalgnn": "cemobam",
}


def normalize_model_name(model_name):
    value = "fedalmer" if model_name is None else str(model_name).strip().lower()
    key = value.replace("-", "").replace("_", "")
    normalized = MODEL_ALIASES.get(key)
    if normalized is None:
        raise ValueError(
            f"Unsupported model_name '{model_name}'. "
            f"Supported: {', '.join(MODEL_CHOICES)}"
        )
    return normalized


def get_model_display_name(model_name):
    return MODEL_DISPLAY_NAMES[normalize_model_name(model_name)]


def build_model(model_name, num_classes, text_input_dim=768, audio_input_dim=768, **model_kwargs):
    normalized_name = normalize_model_name(model_name)

    if normalized_name == "fedalmer":
        from src.architecture.FedalMER import FedalMER

        model = FedalMER(
            text_input_dim=text_input_dim,
            audio_input_dim=audio_input_dim,
            fusion_dim=512,
            projection_dim=256,
            num_heads=4,
            dropout=0.3,
            linear_layer_dims=[512, 256],
            num_classes=num_classes,
        )
    elif normalized_name == "memocmt":
        from src.baseline.MemoCMT.Model import Config as MemoCMTConfig
        from src.baseline.MemoCMT.Model import MemoCMT

        cfg = MemoCMTConfig(
            text_encoder_dim=text_input_dim,
            audio_encoder_dim=audio_input_dim,
            num_attention_head=4,
            dropout=0.3,
            fusion_dim=512,
            linear_layer_output=[512, 256],
            num_classes=num_classes,
            fusion_head_output_type="mean",
        )
        model = MemoCMT(cfg)
    elif normalized_name == "fleser":
        from src.baseline.FleSER.Model import FlexibleMMSER

        model = FlexibleMMSER(
            text_input_dim=text_input_dim,
            audio_input_dim=audio_input_dim,
            num_classes=num_classes,
            fusion_method="self_attention",
            alpha=0.5,
            dropout_rate=0.3,
            use_layernorm=True,
            text_only=False,
            hidden_dim=512,
            proj_dim=256,
            num_heads=4,
        )
    elif normalized_name == "threemser":
        from src.baseline.ThreeMSER.Model import ThreeMSER

        model = ThreeMSER(
            text_input_dim=text_input_dim,
            audio_input_dim=audio_input_dim,
            num_classes=num_classes,
            num_attention_head=8,
            dropout=0.3,
            fusion_dim=128,
        )
    elif normalized_name == "cemobam":
        try:
            from src.baseline.CemoBAM.Model import MultiModalGNN
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "CemoBAM requires optional dependencies (e.g., torch_geometric). "
                "Install them before using model_name='cemobam'."
            ) from exc

        model = MultiModalGNN(
            text_input_dim=text_input_dim,
            audio_input_dim=audio_input_dim,
            hidden_dim=512,
            num_classes=num_classes,
            dropout=0.3,
            heads=4,
            num_layers=3,
            fusion_head_output_type="mean",
            k_text=5,
            k_audio=5,
        )
    else:
        raise ValueError(f"Unsupported model_name '{model_name}'")

    model.num_classes = num_classes
    return model
