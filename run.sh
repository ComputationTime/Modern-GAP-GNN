touch ./results.txt
for i in {1..5}
do
    # python3 ./Experiment.py pma reddit 1 >> ./results.txt
    # python3 ./Experiment.py pma reddit 2 >> ./results.txt
    # python3 ./Experiment.py pma reddit 4 >> ./results.txt
    # python3 ./Experiment.py pma reddit 8 >> ./results.txt
    python3 ./Experiment.py pma facebook 1 >> ./results.txt
    # python3 ./Experiment.py pma facebook 2 >> ./results.txt
    # python3 ./Experiment.py pma facebook 4 >> ./results.txt
    # python3 ./Experiment.py pma facebook 8 >> ./results.txt
    python3 ./Experiment.py pma amazon 1 >> ./results.txt
    python3 ./Experiment.py pma amazon 2 >> ./results.txt
    # python3 ./Experiment.py pma amazon 4 >> ./results.txt
    # python3 ./Experiment.py pma amazon 8 >> ./results.txt
    # python3 ./Experiment.py pmwa reddit 1 >> ./results.txt
    # python3 ./Experiment.py pmwa reddit 2 >> ./results.txt
    # python3 ./Experiment.py pmwa reddit 4 >> ./results.txt
    # python3 ./Experiment.py pmwa reddit 8 >> ./results.txt
    python3 ./Experiment.py pmwa facebook 1 >> ./results.txt
    # python3 ./Experiment.py pmwa facebook 2 >> ./results.txt
    # python3 ./Experiment.py pmwa facebook 4 >> ./results.txt
    # python3 ./Experiment.py pmwa facebook 8 >> ./results.txt
    python3 ./Experiment.py pmwa amazon 1 >> ./results.txt
    python3 ./Experiment.py pmwa amazon 2 >> ./results.txt
    # python3 ./Experiment.py pmwa amazon 4 >> ./results.txt
    # python3 ./Experiment.py pmwa amazon 8 >> ./results.txt
    # python3 ./Experiment.py pmat reddit 1 >> ./results.txt
    # python3 ./Experiment.py pmat reddit 2 >> ./results.txt
    # python3 ./Experiment.py pmat reddit 4 >> ./results.txt
    # python3 ./Experiment.py pmat reddit 8 >> ./results.txt
    python3 ./Experiment.py pmat facebook 1 >> ./results.txt
    # python3 ./Experiment.py pmat facebook 2 >> ./results.txt
    # python3 ./Experiment.py pmat facebook 4 >> ./results.txt
    # python3 ./Experiment.py pmat facebook 8 >> ./results.txt
    python3 ./Experiment.py pmat amazon 1 >> ./results.txt
    python3 ./Experiment.py pmat amazon 2 >> ./results.txt
    # python3 ./Experiment.py pmat amazon 4 >> ./results.txt
    # python3 ./Experiment.py pmat amazon 8 >> ./results.txt
done