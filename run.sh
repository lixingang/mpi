# (
#     python run.py --seed 10 --gpu 0; 
#     python run.py --seed 20 --gpu 0; \
#     python run.py --seed 30 --gpu 0; echo "gpu-0 Done"; 
# ) & sleep 2s

# (
#     python run.py --seed 40 --gpu 1; \
#     python run.py --seed 50 --gpu 1; \
#     python run.py --seed 60 --gpu 1; echo "gpu-1 Done"; 
# ) & sleep 2s

# (
#     python run.py --seed 70 --gpu 2; \
#     python run.py --seed 80 --gpu 2; \
#     python run.py --seed 90 --gpu 2; echo "gpu-2 Done"; 
# ) & sleep 2s

# wait

# echo "END"

echo "START"
# python run.py --seed=10 --gpu=2 --model_name=mlp_v3 --data_dir=Data/v3  & sleep 2s
# python run.py --seed=20 --gpu=0 --model_name=mlp_v3 --data_dir=Data/v3  & sleep 2s
# python run.py --seed=30 --gpu=1 --model_name=mlp_v3 --data_dir=Data/v3  & sleep 2s
# python run.py --seed=40 --gpu=2 --model_name=mlp_v3 --data_dir=Data/v3  & sleep 2s
# wait
# python run.py --seed=50 --gpu=2 --model_name=mlp_v3 --data_dir=Data/v3  & sleep 2s
# python run.py --seed=60 --gpu=2 --model_name=mlp_v3 --data_dir=Data/v3  & sleep 2s
# python run.py --seed=70 --gpu=1 --model_name=mlp_v3 --data_dir=Data/v3  & sleep 2s
# python run.py --seed=80 --gpu=0 --model_name=mlp_v3 --data_dir=Data/v3  & sleep 2s
# wait
# python run.py --seed=90 --gpu=2 --model_name=mlp_v3 --data_dir=Data/v3  & sleep 2s
# python run.py --seed=100 --gpu=2 --model_name=mlp_v3 --data_dir=Data/v3 & sleep 2s
# wait

python run.py --seed 10  --gpu 2 --model_name=gp_v3 --data_dir=Data/v4  & sleep 2s
python run.py --seed=20  --gpu=1 --model_name=gp_v3 --data_dir=Data/v4  & sleep 2s
python run.py --seed=30  --gpu=0 --model_name=gp_v3 --data_dir=Data/v4  & sleep 2s
python run.py --seed=40  --gpu=2 --model_name=gp_v3 --data_dir=Data/v4  & sleep 2s
wait
python run.py --seed=50  --gpu=2 --model_name=gp_v3 --data_dir=Data/v4  & sleep 2s
python run.py --seed=60  --gpu=2 --model_name=gp_v3 --data_dir=Data/v4  & sleep 2s
python run.py --seed=70  --gpu=1 --model_name=gp_v3 --data_dir=Data/v4  & sleep 2s
python run.py --seed=80  --gpu=0 --model_name=gp_v3 --data_dir=Data/v4  & sleep 2s
wait
python run.py --seed=90  --gpu=2 --model_name=gp_v3 --data_dir=Data/v4  & sleep 2s
python run.py --seed=100 --gpu=2 --model_name=gp_v3 --data_dir=Data/v4  & sleep 2s
wait
echo "END"