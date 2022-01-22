action(0):: action(no_op);      % 0
action(1):: action(push_up);    % 1
action(2):: action(push_down);  % 2
action(3):: action(push_left);  % 3
action(4):: action(push_right). % 4

box(0):: box( 0, 1). % 5
box(1):: box(-1, 0). % 9
box(2):: box( 1, 0). % 10
box(3):: box( 0,-1). % 14

corner(0):: corner( 0, 2). % 2
corner(1):: corner(-2, 0). % 8
corner(2):: corner( 2, 0). % 11
corner(3):: corner( 0,-2). % 17


box_transition( X,  Y, no_op,       X,  Y).
box_transition(-1,  0, push_left,  -2,  0).
box_transition( 1,  0, push_right,  2,  0).
box_transition( 0,  1, push_up,     0,  2).
box_transition( 0, -1, push_down,   0, -2).
box_transition( X,  Y, push_left,   X,  Y):- \+(X =:= -1, Y =:= 0).
box_transition( X,  Y, push_right,  X,  Y):- \+(X =:=  1, Y =:= 0).
box_transition( X,  Y, push_up,     X,  Y):- \+(X =:=  0, Y =:= 1).
box_transition( X,  Y, push_down,   X,  Y):- \+(X =:=  0, Y =:= -1).


unsafe_next :-
    box( X,  Y),
    action(A),
    box_transition(X, Y, A, NextX, NextY),
    corner(NextX, NextY).


safe_next:- \+unsafe_next.
safe_action(A):- action(A), safe_next.
