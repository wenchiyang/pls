action(0)::action(stay);
action(1)::action(up);
action(2)::action(down);
action(3)::action(left);
action(4)::action(right).

sensor_value(0)::ghost(up).
sensor_value(1)::ghost(down).
sensor_value(2)::ghost(left).
sensor_value(3)::ghost(right).

% transition(Action, NextPos)
transition(stay,here).
transition(left,left).
transition(right,right).
transition(up,up).
transition(down,down).


unsafe_next :- action(A), transition(A,NextPos), ghost(NextPos).
safe_next :- \+ unsafe_next.
