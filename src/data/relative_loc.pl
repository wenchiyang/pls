action(0)::action(stay);
action(1)::action(up);
action(2)::action(down);
action(3)::action(left);
action(4)::action(right).

ghost(0)::ghost(up).
ghost(1)::ghost(down).
ghost(2)::ghost(left).
ghost(3)::ghost(right).

wall(0)::wall(up).
wall(1)::wall(down).
wall(2)::wall(left).
wall(3)::wall(right).


% transition(Action, NextPos)
transition(stay,here).
transition(left,left).
transition(right,right).
transition(up,up).
transition(down,down).


transition_with_wall(A,here) :-
    transition(A,NextPos), wall(NextPos).
transition_with_wall(A,NextPos) :-
    transition(A,NextPos), \+ wall(NextPos).


unsafe_next :- action(A), transition_with_wall(A,NextPos), ghost(NextPos).
safe_next :- \+ unsafe_next.

safe_action(A):- action(A), safe_next.