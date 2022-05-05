echo "START"

#---
TAG=fds0505
MODEL=mlp
echo ${MODEL}_${TAG}
# python run.py --seed=10  --gpu=0 --tag=${TAG} --model=${MODEL} & sleep 1s

# python run.py --seed=20  --gpu=1 --tag=${TAG} --model=${MODEL} & sleep 1s

# python run.py --seed=30  --gpu=2 --tag=${TAG} --model=${MODEL} & sleep 1s

# # wait

# python run.py --seed=40  --gpu=0 --tag=${TAG} --model=${MODEL}  & sleep 1s

# python run.py --seed=50  --gpu=1 --tag=${TAG} --model=${MODEL}  & sleep 1s

# python run.py --seed=60  --gpu=2 --tag=${TAG} --model=${MODEL}  & sleep 1s

# wait
python get_logs.py --name=v1_${MODEL}_${TAG}



# python run.py --seed=10  --gpu=0   --tag=0505_fds  --model=mlp  