# echo "START"
# python main.py get_list Config/config.yaml &
# wait
# CUDA_VISIBLE_DEVICES=1 python main.py 0 Config/config.yaml &
# CUDA_VISIBLE_DEVICES=2 python main.py 1 Config/config.yaml &
# CUDA_VISIBLE_DEVICES=0 python main.py 2 Config/config.yaml &
# CUDA_VISIBLE_DEVICES=1 python main.py 3 Config/config.yaml &
# CUDA_VISIBLE_DEVICES=2 python main.py 4 Config/config.yaml &
# wait
# echo "END"

# python post_analysis.py --name swint_config_baseline

CONFIG=swint192.yaml 
TAG=lds

python main.py Config/${CONFIG} ${TAG}
python post_analysis get_logs ${CONFIG}_${TAG}