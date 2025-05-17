folder="$(realpath $(dirname $0))"

for task in "$folder/"*.task; do
    echo "Extracting $task."
    unzip -n "$task" -d "$folder"
done

pip install --break-system-packages --upgrade tf2onnx
for model in "$folder/"*.tflite; do
    output="${model%.tflite}.onnx"
    if [ -f "$output" ]; then
        echo "Skipping $model."
        continue
    else
        echo "Converting $model."
        python -m tf2onnx.convert \
            --tflite "$model" \
            --output "${model%.tflite}.onnx" \
            --opset 13
    fi
done