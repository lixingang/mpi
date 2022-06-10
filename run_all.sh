TAG=$1

python main.py run_all Config/swint.yaml ${TAG}
python post_analysis.py get_logs swint_${TAG}