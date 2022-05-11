echo "START"

#---
TAG=both_loss
MODEL=mlp
# echo ${MODEL}_${TAG}
python main.py --seed=10  --gpu=0 --tag=${TAG} --model=${MODEL} & sleep 1s

python main.py --seed=20  --gpu=1 --tag=${TAG} --model=${MODEL} & sleep 1s

python main.py --seed=30  --gpu=2 --tag=${TAG} --model=${MODEL} & sleep 1s

python main.py --seed=40  --gpu=0 --tag=${TAG} --model=${MODEL}  & sleep 1s

python main.py --seed=50  --gpu=1 --tag=${TAG} --model=${MODEL}  & sleep 1s


python main.py --seed=60  --gpu=0 --tag=${TAG} --model=${MODEL}  & sleep 1s

python main.py --seed=70  --gpu=0 --tag=${TAG} --model=${MODEL}  & sleep 1s

python main.py --seed=80  --gpu=2 --tag=${TAG} --model=${MODEL}  & sleep 1s

python main.py --seed=90  --gpu=2 --tag=${TAG} --model=${MODEL}  & sleep 1s

python main.py --seed=100  --gpu=1 --tag=${TAG} --model=${MODEL}  & sleep 1s

wait
python get_logs.py --name=${MODEL}_${TAG}



# python main.py --seed=10  --gpu=0   --tag=fds0508  --model=mlp  