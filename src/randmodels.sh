#/bin/sh
# runs random test

for i in 100 1000 10000 100000 1000000
do
  python randgame.py $i | tail -n 1 >> randresults.txt
done
