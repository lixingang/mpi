CFG=swint
TAG=$1

python main.py run_1_fold Config/${CFG}.yaml ${TAG} 5
# python post_analysis.py get_logs ${CFG}_${TAG}