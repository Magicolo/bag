from pathlib import Path
import zipfile
from tf2onnx.convert import from_function
import tensorflow

for file in Path(__file__).parent.glob("*.task"):
    print(f"=> Extracting '{file}'.")
    with zipfile.ZipFile(file) as zip:
        zip.extractall(file.parent)

for file in Path(__file__).parent.glob("*.tflite"):
    print(f"=> Converting '{file}'.")
    model = tensorflow.saved_model.load(file)
    concrete = model.signatures["serving_default"]
    name, settings = next(concrete.structured_input_signature[1].items())
    settings.shape[0] = None
    specification = tensorflow.TensorSpec(settings.shape, settings.dtype, name)
    from_function(
        concrete,
        input_signature=[specification],
        output_path=file.with_suffix(".onnx"),
        opset=13,
    )
