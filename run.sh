echo "START"

#---
TAG=bothloss_lessdata
MODEL=mlp
echo ${MODEL}_${TAG}
python main.py --seed=10  --gpu=0 --tag=${TAG} --model=${MODEL} & sleep 1s

python main.py --seed=20  --gpu=0 --tag=${TAG} --model=${MODEL} & sleep 1s

python main.py --seed=30  --gpu=1 --tag=${TAG} --model=${MODEL} & sleep 1s

python main.py --seed=40  --gpu=1 --tag=${TAG} --model=${MODEL}  & sleep 1s

python main.py --seed=50  --gpu=2 --tag=${TAG} --model=${MODEL}  & sleep 1s


python main.py --seed=60  --gpu=2 --tag=${TAG} --model=${MODEL}  & sleep 1s

# python main.py --seed=70  --gpu=2 --tag=${TAG} --model=${MODEL}  & sleep 1s

# python main.py --seed=80  --gpu=2 --tag=${TAG} --model=${MODEL}  & sleep 1s

# python main.py --seed=90  --gpu=2 --tag=${TAG} --model=${MODEL}  & sleep 1s

wait
python post_analysis.py --name=${MODEL}_${TAG}



# python main.py --seed=10  --gpu=0   --tag=test  --model=mlp  

# 预测
# files=$(ls -d /home/lxg/data/mpi/Logs/v1_mlp_0428/*/)
# for file in $files
# do
#     echo predicting $file ...
#     python predict.py --log_dir=$file
# done