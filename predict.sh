files=$(ls -d Logs/mpi/*/)
for file in $files
do
    echo predicting $file ...
    python predict.py --log_dir=$file
done