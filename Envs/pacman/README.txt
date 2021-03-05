The implementation builds on the Pac-Man projects developed for UC Berkeley's introductory artificial intelligence course, CS 188.
http://ai.berkeley.edu/project_overview.html
The Pac-Man projects are written in pure Python 3 and do not depend on any packages external to a standard Python distribution.
Only to compute the shields (if one does not want to use the precomputed shields) Stormpy is required. https://github.com/moves-rwth/stormpy


----------------------------------------------------
Load pre-computed shields and apply the Shield to RL
----------------------------------------------------
The provided files include precomputed shields (.dump files) for all grids contained in the folder /layouts. 
To reproduce the examples from the paper, use the commands:

python pacman.py -o shields/smallGrid.dump -p ApproximateQAgent -a extractor=SimpleExtractor -g DirectionalGhost -x 0 -y 50 -n 70 -l smallGrid.lay 
python pacman.py -o shields/smallGrid2.dump -p ApproximateQAgent -a extractor=SimpleExtractor -g DirectionalGhost -x 0 -y 50 -n 70 -l smallGrid2.lay 
etc.

-x ... number of training episodes without shield 
-y ... number of training episodes with shield (learning is started new)
-n ... total number of played games (i.e., n-x-y ... number of games in exploitation phase)

If a labyrinth is symmetric on the x-axis (y-axis), please use additionally the option --symX (--symY respectively)

------------------------
Compute + Apply shield
------------------------
To compute the shields for yourself and to apply them, use the following commands: 
(Note: Computing the shields takes some time)



python pacman.py -p ApproximateQAgent -g DirectionalGhost -a extractor=SimpleExtractor -x 50 -y 50 -n 110 -l smallGrid.lay
python pacman.py -p ApproximateQAgent -g DirectionalGhost -a extractor=SimpleExtractor -x 50 -y 50 -n 110 -l smallGrid2.lay
etc.

If a labyrinth is symmetric on the x-axis (y-axis), please use additionally the option --symX (--symY respectively)


 
