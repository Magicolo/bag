folder=$(realpath "$(dirname $0)")
models="https://storage.googleapis.com/mediapipe-models"
curl --location --output-dir "$folder" --remote-name "$models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task" &
curl --location --output-dir "$folder" --remote-name "$models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" &
curl --location --output-dir "$folder" --remote-name "$models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task" &
curl --location --output-dir "$folder" --remote-name "$models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task" &
curl --location --output-dir "$folder" --remote-name "$models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task" &
curl --location --output-dir "$folder" --remote-name "$models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" &
curl --location --output-dir "$folder" --remote-name "$models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task" &
curl --location --output-dir "$folder" --remote-name "$models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite" &
curl --location --output-dir "$folder" --remote-name "$models/image_segmenter/hair_segmenter/float32/latest/hair_segmenter.tflite" &
curl --location --output-dir "$folder" --remote-name "$models/image_segmenter/selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite" &
curl --location --output-dir "$folder" --remote-name "$models/image_segmenter/deeplab_v3/float32/latest/deeplab_v3.tflite" &
curl --location --output-dir "$folder" --remote-name "$models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite" &
curl --location --output-dir "$folder" --remote-name "$models/interactive_segmenter/magic_touch/float32/latest/magic_touch.tflite" &
wait