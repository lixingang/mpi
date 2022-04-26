echo "START"
TAG=mlp_gp
DATA=v1
echo $DATA_DIR_$TAG
python run.py --seed 10  --gpu=0 --data=$DATA --tag=$TAG --use_gp=1 & sleep 2s
python run.py --seed=20  --gpu=0 --data=$DATA --tag=$TAG --use_gp=1 & sleep 2s
python run.py --seed=30  --gpu=1 --data=$DATA --tag=$TAG --use_gp=1 & sleep 2s
python run.py --seed=40  --gpu=1 --data=$DATA --tag=$TAG --use_gp=1 & sleep 2s
python run.py --seed=50  --gpu=2 --data=$DATA --tag=$TAG --use_gp=1 & sleep 2s
python run.py --seed=60  --gpu=2 --data=$DATA --tag=$TAG --use_gp=1 & sleep 2s
wait
python get_logs.py --name=${DATA}_${TAG}
echo "END"

TAG=mlp
DATA=v1
echo $DATA_DIR_$TAG
python run.py --seed 10  --gpu=0 --data=$DATA --tag=$TAG --use_gp=0 & sleep 2s
python run.py --seed=20  --gpu=0 --data=$DATA --tag=$TAG --use_gp=0 & sleep 2s
python run.py --seed=30  --gpu=1 --data=$DATA --tag=$TAG --use_gp=0 & sleep 2s
python run.py --seed=40  --gpu=1 --data=$DATA --tag=$TAG --use_gp=0 & sleep 2s
python run.py --seed=50  --gpu=2 --data=$DATA --tag=$TAG --use_gp=0 & sleep 2s
python run.py --seed=60  --gpu=2 --data=$DATA --tag=$TAG --use_gp=0 & sleep 2s
wait
python get_logs.py --name=${DATA}_${TAG}
echo "END"


