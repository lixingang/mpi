echo "START"
python main.py get_list Config/config.yaml &
wait
CUDA_VISIBLE_DEVICES=1 python main.py run 0 Config/config.yaml &
CUDA_VISIBLE_DEVICES=2 python main.py run 1 Config/config.yaml &
CUDA_VISIBLE_DEVICES=0 python main.py run 2 Config/config.yaml &
wait
CUDA_VISIBLE_DEVICES=1 python main.py run 3 Config/config.yaml &
CUDA_VISIBLE_DEVICES=2 python main.py run 4 Config/config.yaml &
wait
echo "END"

# python post_analysis.py --name swint_config_baseline