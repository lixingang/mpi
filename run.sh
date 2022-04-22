echo "START"

MODEL_NAME=gp_v3
DATA_DIR=Data/v3
echo $MODEL_NAME
echo $DATA_DIR
wait
python run.py --seed 10  --gpu 2 --model_name=$MODEL_NAME --data_dir=$DATA_DIR & sleep 2s
python run.py --seed=20  --gpu=1 --model_name=$MODEL_NAME --data_dir=$DATA_DIR & sleep 2s
python run.py --seed=30  --gpu=0 --model_name=$MODEL_NAME --data_dir=$DATA_DIR & sleep 2s
python run.py --seed=40  --gpu=2 --model_name=$MODEL_NAME --data_dir=$DATA_DIR & sleep 2s
wait 
python run.py --seed=50  --gpu=2 --model_name=$MODEL_NAME --data_dir=$DATA_DIR & sleep 2s
python run.py --seed=60  --gpu=2 --model_name=$MODEL_NAME --data_dir=$DATA_DIR & sleep 2s
python run.py --seed=70  --gpu=1 --model_name=$MODEL_NAME --data_dir=$DATA_DIR & sleep 2s
python run.py --seed=80  --gpu=0 --model_name=$MODEL_NAME --data_dir=$DATA_DIR & sleep 2s
wait 
python run.py --seed=90  --gpu=2 --model_name=$MODEL_NAME --data_dir=$DATA_DIR & sleep 2s
python run.py --seed=100 --gpu=2 --model_name=$MODEL_NAME --data_dir=$DATA_DIR & sleep 2s
wait


#########
python get_logs.py --log_dir=$MODEL_NAME
echo "END"