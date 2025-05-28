folder=$(realpath "$(dirname $0)")

nice -n -20 ionice -c1 -n0 chrt -f 99 python "$folder/main.py"
