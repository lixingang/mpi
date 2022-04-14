# (
#     python train.py --seed 10 --gpu 0; 
#     python train.py --seed 20 --gpu 0; \
#     python train.py --seed 30 --gpu 0; echo "gpu-0 Done"; 
# ) & sleep 2s

# (
#     python train.py --seed 40 --gpu 1; \
#     python train.py --seed 50 --gpu 1; \
#     python train.py --seed 60 --gpu 1; echo "gpu-1 Done"; 
# ) & sleep 2s

# (
#     python train.py --seed 70 --gpu 2; \
#     python train.py --seed 80 --gpu 2; \
#     python train.py --seed 90 --gpu 2; echo "gpu-2 Done"; 
# ) & sleep 1s

# wait

# echo "END"

echo "START"
python train.py --seed 1000 --gpu 0 & sleep 1s
python train.py --seed 2000 --gpu 0 & sleep 1s
python train.py --seed 3000 --gpu 2 & sleep 1s
python train.py --seed 4000 --gpu 2 & sleep 1s
python train.py --seed 5000 --gpu 2 & sleep 1s
python train.py --seed 6000 --gpu 0 & sleep 1s
python train.py --seed 7000 --gpu 0 & sleep 1s
python train.py --seed 8000 --gpu 2 & sleep 1s
python train.py --seed 9000 --gpu 2 & sleep 1s
python train.py --seed 10000 --gpu 2 & sleep 1s
wait
echo "END"