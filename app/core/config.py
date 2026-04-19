from pathlib import Path
import onnxruntime as ort

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

MRI_MODEL_PATH = MODEL_DIR / "model_mri.onnx"
EEG_MODEL_PATH = MODEL_DIR / "model_eeg.onnx"


def create_onnx_session(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"]
    )
    return session


mri_session = create_onnx_session(MRI_MODEL_PATH)
eeg_session = create_onnx_session(EEG_MODEL_PATH)


def get_model_io_details(session: ort.InferenceSession):
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]
    input_shapes = [inp.shape for inp in session.get_inputs()]
    output_shapes = [out.shape for out in session.get_outputs()]
    input_types = [inp.type for inp in session.get_inputs()]
    output_types = [out.type for out in session.get_outputs()]

    return {
        "input_names": input_names,
        "output_names": output_names,
        "input_shapes": input_shapes,
        "output_shapes": output_shapes,
        "input_types": input_types,
        "output_types": output_types,
    }