CFG=swint
TAG=$1
FOLD=$2
python main.py run_1_fold ${CFG}.yaml ${TAG} ${FOLD}

# python post_analysis.py get_logs ${CFG}_${TAG}