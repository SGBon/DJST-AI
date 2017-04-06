#/bin/sh
# creates training models for increasing number of runs
# and outputs them to folder
MODEL_DIR=$1

if [ $# -lt 1 ]
then
  echo "USAGE: $0 [model folder]"
  exit 1
fi

> trainresults.txt # clear the file before appending

for i in 100 1000 10000 100000 1000000
do
  python traingame.py $i $MODEL_DIR/train$i.tr | tail -n 1 >> trainresults.txt
done
