TAG=$1

python main.py run_all swint.yaml ${TAG}
python main.py get_logs swint_${TAG}