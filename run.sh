folder=$(realpath "$(dirname $0)")

pactl load-module module-alsa-sink device=hw:4,0 sink_name=usb7d
python "$folder/code/main.py"
