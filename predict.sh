files=$(ls -d Logs/mpi_full/*/)
for file in $files
do
    echo predicting $file ...
    python predict.py --log_dir=$file
done