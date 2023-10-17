python3 ~/autogoal/docs/shell/printshell.py autogoal
python3 -m autogoal

sleep 2

python3 ~/autogoal/docs/shell/printshell.py autogoal ml fit dataset.csv
python3 -m autogoal ml fit ~/autogoal/autogoal/datasets/data/uci_cars/car.data --format csv --iterations 3 --pop-size 3 --random-state 1

sleep 3

python3 ~/autogoal/docs/shell/printshell.py autogoal ml inspect
python3 -m autogoal ml inspect

sleep 3

python3 ~/autogoal/docs/shell/printshell.py autogoal ml predict new.csv
python3 -m autogoal ml predict ~/autogoal/autogoal/datasets/data/uci_cars/car.data --format csv --ignore-cols -1

sleep 5
