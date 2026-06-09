parent(tom, bob).
parent(pam, bob).
parent(tom, bob).
parent(tom, liz).
parent(bob, ann).
parent(bob, pat).
parent(mary, ann).
parent(pat, juli).

like(bob, pam).

child(Y, X):-parent(X, Y).

male(tom).
male(bob).
male(jim).
female(liz).
female(pam).
female(pat).
female(ann).

mother(X, Y):-parent(X, Y),female(X).

different(X,Y):- X \= Y.
sister(X, Y):- parent(Z,X),parent(Z,Y),female(X),different(X, Y).