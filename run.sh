python train.py --seed 10 --gpu 0 & sleep 1s
python train.py --seed 20 --gpu 1 & sleep 1s
wait
python train.py --seed 30 --gpu 0 & sleep 1s
python train.py --seed 40 --gpu 1 & sleep 1s
wait
python train.py --seed 50 --gpu 0 & sleep 1s
python train.py --seed 60 --gpu 1 & sleep 1s
wait
python train.py --seed 70 --gpu 0 & sleep 1s
python train.py --seed 80 --gpu 1 & sleep 1s
wait
python train.py --seed 90 --gpu 0 & sleep 1s
python train.py --seed 100 --gpu 1 & sleep 1s
echo "END"