from pathlib import Path
import zipfile
from tf2onnx.convert import from_tflite

for file in Path(__file__).parent.glob("*.task"):
    print(f"=> Extracting '{file}'.")
    with zipfile.ZipFile(file) as zip:
        zip.extractall(file.parent)

for file in Path(__file__).parent.glob("*.tflite"):
    output = file.with_suffix(".onnx")
    if output.exists():
        print(f"=> Skipping '{file}' as '{output}' already exists.")
        continue

    print(f"=> Converting '{file}' with settings.")
    try:
        from_tflite(file, output_path=output, opset=13)
        print(f"=> Succeeded to convert '{file}' to '{output}'.")
    except Exception as error:
        print(f"=> Failed to convert '{file}' with '{error}'.")
