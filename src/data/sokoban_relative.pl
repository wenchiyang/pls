action(0):: action(no_op);      % 0
action(1):: action(push_up);    % 1
action(2):: action(push_down);  % 2
action(3):: action(push_left);  % 3
action(4):: action(push_right); % 4
action(5):: action(move_up);    % 5
action(6):: action(move_down);  % 6
action(7):: action(move_left);  % 7
action(8):: action(move_right). % 8

box(0):: box( 0, 1). % 5
box(1):: box(-1, 0). % 9
box(2):: box( 1, 0). % 10
box(3):: box( 0,-1). % 14

wall( 0):: wall( 0, 3). % 0
wall( 1):: wall(-1, 2). % 1
wall( 2):: wall( 0, 2). % 2
wall( 3):: wall( 1, 2). % 3
wall( 4):: wall(-2, 1). % 4
wall( 5):: wall(-1, 1). % 5
wall( 6):: wall( 0, 1). % 6
wall( 7):: wall( 1, 1). % 7
wall( 8):: wall( 2, 1). % 8
wall( 9):: wall(-3, 0). % 9
wall(10):: wall(-2, 0). % 10
wall(11):: wall(-1, 0). % 11
wall(12):: wall( 1, 0). % 12
wall(13):: wall( 2, 0). % 13
wall(14):: wall( 3, 0). % 14
wall(15):: wall(-2,-1). % 15
wall(16):: wall(-1,-1). % 16
wall(17):: wall( 0,-1). % 17
wall(18):: wall( 1,-1). % 18
wall(19):: wall( 2,-1). % 19

target(0):: target( 0, 2). % 2
target(1):: target(-2, 0). % 8
target(2):: target( 2, 0). % 11
target(3):: target( 0,-2). % 17

% transition(Action, NextPos)
%transition(action(no_op),      0,  0).
%transition(action(move_left), -1,  0).
%transition(action(move_right), 1,  0).
%transition(action(move_up),    0,  1).
%transition(action(move_down),  0, -1).
%transition(action(push_left), -1,  0).
%transition(action(push_right), 1,  0).
%transition(action(push_up),    0,  1).
%transition(action(push_down),  0, -1).
%
%transition_with_wall(action(A), NextX, NextY):-
%    transition(action(A), NextX, NextY),
%    \+wall(NextX, NextY),
%    \+box(NextX, NextY).
%
%transition_with_wall(action(A), 0, 0):-
%    transition(action(A), NextX, NextY),
%    wall(NextX, NextY).
%
%transition_with_wall(action(A), 0, 0):-
%    transition(action(A), NextX, NextY),
%    box(NextX, NextY),
%    has_neighboring_wall(NextX, NextY, A).
%
%transition_with_wall(action(A), 0, 0):-
%    transition(action(A), NextX, NextY),
%    box(NextX, NextY),
%    has_neighboring_box(NextX, NextY, A).

box_transition( X,  Y, action(no_op),       X,  Y):- box(X, Y).
box_transition( X,  Y, action(move_left),   X,  Y):- box(X, Y).
box_transition( X,  Y, action(move_right),  X,  Y):- box(X, Y).
box_transition( X,  Y, action(move_up),     X,  Y):- box(X, Y).
box_transition( X,  Y, action(move_down),   X,  Y):- box(X, Y).
box_transition(-1,  0, action(push_left),  -2,  0):- box(-1,  0).
box_transition( 1,  0, action(push_right),  2,  0):- box( 1,  0).
box_transition( 0,  1, action(push_up),     0,  2):- box( 0,  1).
box_transition( 0, -1, action(push_down),   0, -2):- box( 0, -1).
box_transition( X,  Y, action(push_left),   X,  Y):-
    box( X,  Y), \+(X =:= -1, Y =:= 0).
box_transition( X,  Y, action(push_right),  X,  Y):-
    box( X,  Y), \+(X =:=  1, Y =:= 0).
box_transition( X,  Y, action(push_up),     X,  Y):-
    box( X,  Y), \+(X =:=  0, Y =:= 1).
box_transition( X,  Y, action(push_down),   X,  Y):-
    box( X,  Y), \+(X =:=  0, Y =:= -1).

box_transition_with_wall(X, Y, action(A), NextX, NextY):-
    box_transition(X, Y, action(A), NextX, NextY),
    \+wall(NextX, NextY),
    \+box(NextX, NextY).

box_transition_with_wall(X, Y, action(A), X, Y):-
    box_transition(X, Y, action(A), NextX, NextY),
    (wall(NextX, NextY); box(NextX, NextY)).

has_neighboring_wall(X, Y, up):-    Y1 is Y+1, wall(X, Y1).
has_neighboring_wall(X, Y, down):-  Y1 is Y-1, wall(X, Y1).
has_neighboring_wall(X, Y, left):-  X1 is X-1, wall(X1, Y).
has_neighboring_wall(X, Y, right):- X1 is X+1, wall(X1, Y).
has_neighboring_box(X, Y, up):-    Y1 is Y+1, box(X, Y1).
has_neighboring_box(X, Y, down):-  Y1 is Y-1, box(X, Y1).
has_neighboring_box(X, Y, left):-  X1 is X-1, box(X1, Y).
has_neighboring_box(X, Y, right):- X1 is X+1, box(X1, Y).

neighboring_wall_number(X, Y, N):-
    findall(Dir, has_neighboring_wall(X, Y, Dir), B),
    length(B, N).

corner(X, Y):-
    \+wall(X, Y),
    neighboring_wall_number(X, Y, N),
    N > 1.

unsafe_next:-
    box(X, Y),
    action(A),
    box_transition_with_wall(X, Y, action(A), NextX, NextY),
    corner(NextX, NextY), \+target(NextX, NextY).

safe_next:- \+unsafe_next.
safe_action(A):- action(A), safe_next.
