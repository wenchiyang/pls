The implementation builds on the Pac-Man projects developed for UC Berkeley's introductory artificial intelligence course, CS 188.
http://ai.berkeley.edu/project_overview.html
The Pac-Man projects are written in pure Python 3 and do not depend on any packages external to a standard Python distribution.
Only to compute the shields (if one does not want to use the precomputed shields) Stormpy is required. https://github.com/moves-rwth/stormpy


----------------------------------------------------
Load pre-computed shields and apply the Shield to RL
----------------------------------------------------
The provided files include precomputed shields (.dump files) for the warehouse layout contained in the folder /layouts. The shields differ, on how many crossings are shielded (2,4,8 crossings).
To load them and to reproduce the examples from the paper, use the commands:

python warehouse.py -o shields/warehouse_2_crossings.dump -b 5 -p ApproximateQAgent -g ForkTruckPath -a extractor=SimpleExtractor -x 0 -y 50 -n 70 -l warehouse.lay
python warehouse.py -o shields/warehouse_4_crossings.dump -b 7 -p ApproximateQAgent -g ForkTruckPath -a extractor=SimpleExtractor -x 0 -y 50 -n 70 -l warehouse.lay
python warehouse.py -o shields/warehouse_8_crossings.dump -b 8 -p ApproximateQAgent -g ForkTruckPath -a extractor=SimpleExtractor -x 0 -y 50 -n 70 -l warehouse.lay

-x ... number of training episodes without shield 
-y ... number of training episodes with shield (learning is started new)
-n ... total number of played games (i.e., n-x-y ... number of games in exploitation phase)
-b ... distance to the exit, in which crossigns will be shielded



------------------------
Compute + Apply shield
------------------------
To compute the shields for yourself and to apply them, use the following commands: 
(Note: Computing the shield takes some time)

python warehouse.py -b 5 -p ApproximateQAgent -g ForkTruckPath -a extractor=SimpleExtractor -x 50 -y 50 -n 110 -l warehouse.lay
python warehouse.py -b 7 -p ApproximateQAgent -g ForkTruckPath -a extractor=SimpleExtractor -x 50 -y 50 -n 110 -l warehouse.lay
python warehouse.py -b 8 -p ApproximateQAgent -g ForkTruckPath -a extractor=SimpleExtractor -x 50 -y 50 -n 110 -l warehouse.lay


