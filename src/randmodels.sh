#/bin/sh
# runs random test

> randresults.txt # clear the file before appending

for i in 100 1000 10000 100000 1000000
do
  python3 randgame.py $i | tail -n 1 >> randresults.txt
done
