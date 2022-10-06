% actions
action(0)::action(do_nothing);
action(1)::action(accelerate);
action(2)::action(brake);
action(3)::action(turn_left);
action(4)::action(turn_right).

% states (discretized)
grass(0)::grass(in_front).
grass(1)::grass(on_the_left).
grass(2)::grass(on_the_right).



% transition
unsafe_next :- grass(in_front), action(accelerate).
unsafe_next :- grass(on_the_left), \+ grass(on_the_right), action(turn_left).
unsafe_next :- \+ grass(on_the_left), grass(on_the_right), action(turn_right).

safe_next:- \+unsafe_next.
safe_action(A):- action(A).