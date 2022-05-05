files=$(ls -d /home/lxg/data/mpi/Logs/v1_mlp_0428/*/)
for file in $files
do
    echo predicting $file ...
    python predict.py --log_dir=$file
done