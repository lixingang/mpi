TAG=$1

python main.py run_all swint192.yaml ${TAG}
python predict.py get_logs swint_${TAG}