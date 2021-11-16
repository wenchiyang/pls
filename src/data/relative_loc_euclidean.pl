
close(0,1, -1,1).
close(0,1, 1,1).

close(0,-1, -1,-1).
close(0,-1, 1,-1).

close(-1,0, -1,1).
close(-1,0, -1,-1).

close(1,0, 1,1).
close(1,0, 1,-1).


action(0)::action(stay);
action(1)::action(up);
action(2)::action(down);
action(3)::action(left);
action(4)::action(right).

ghost(0)::ghost(0, 1).
ghost(1)::ghost(0, -1).
ghost(2)::ghost(-1, 0).
ghost(3)::ghost(1, 0).

ghost(4)::ghost(-1, -1). % right up
ghost(5)::ghost(1, 1).
ghost(6)::ghost(1, -1).
ghost(7)::ghost(-1, -1).


wall(0)::wall(0, 1).
wall(1)::wall(0, -1).
wall(2)::wall(-1, 0).
wall(3)::wall(1, 0).

wall(4)::wall(-1, -1). % right up
wall(5)::wall(1, 1).
wall(6)::wall(1, -1).
wall(7)::wall(-1, -1).


% transition(Action, NextPos)
transition(stay,0,0).
transition(left,-1,0).
transition(right,1,0).
transition(up,0,1).
transition(down,0,-1).


transition_with_wall(A, 0, 0) :-
    transition(A, NextX, NextY), wall(NextX, NextY).
transition_with_wall(A,NextX, NextY) :-
    transition(A,NextX, NextY), \+ wall(NextX, NextY).

unsafe_next :-
    action(A),
    transition_with_wall(A, NextX, NextY),
    ghost(NextX, NextY).
unsafe_next :-
    action(A),
    transition_with_wall(A, NextX, NextY),
    close(NextX, NextY, NextX2, NextY2),
    ghost(NextX2, NextY2).

safe_next :- \+ unsafe_next.

safe_action(A):- action(A), safe_next.