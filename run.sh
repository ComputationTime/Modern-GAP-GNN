touch ./results.txt
for i in {1..5}
do
    python3 ./Experiment.py pma reddit 1 >> ./results.txt
    python3 ./Experiment.py pma reddit 2 >> ./results.txt
    python3 ./Experiment.py pma reddit 4 >> ./results.txt
    python3 ./Experiment.py pma reddit 8 >> ./results.txt
    python3 ./Experiment.py pmwa reddit 1 >> ./results.txt
    python3 ./Experiment.py pmwa reddit 2 >> ./results.txt
    python3 ./Experiment.py pmwa reddit 4 >> ./results.txt
    python3 ./Experiment.py pmwa reddit 8 >> ./results.txt
    python3 ./Experiment.py pma reddit 1 >> ./results.txt
    python3 ./Experiment.py pma reddit 2 >> ./results.txt
    python3 ./Experiment.py pma reddit 4 >> ./results.txt
    python3 ./Experiment.py pma reddit 8 >> ./results.txt
    python3 ./Experiment.py pma facebook 1 >> ./results.txt
    python3 ./Experiment.py pma facebook 2 >> ./results.txt
    python3 ./Experiment.py pma facebook 4 >> ./results.txt
    python3 ./Experiment.py pma facebook 8 >> ./results.txt
    python3 ./Experiment.py pmwa facebook 1 >> ./results.txt
    python3 ./Experiment.py pmwa facebook 2 >> ./results.txt
    python3 ./Experiment.py pmwa facebook 4 >> ./results.txt
    python3 ./Experiment.py pmwa facebook 8 >> ./results.txt
done