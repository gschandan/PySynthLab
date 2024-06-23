(set-logic LIA)

(declare-fun f (Int Int) Int)
(define-fun f ((x Int) (y Int)) Int (ite (<= x y) y x))

(declare-fun x () Int)
(declare-fun y () Int)

(assert (= (f x y) (f y x)))
(assert (and (<= x (f x y)) (<= y (f x y))))

(check-sat)
(get-model)

does there exist an x and y such that either 9 OR 10 are false
i.e.
(assert( or (not (= (f x y) (f y x)))
(not (and (<= x (f x y)) (<= y (f x y))))

if (define-fun f ((x Int) (y Int)) Int x ) incorrect -> counterexample


(set-logic LIA)

(declare-fun f (Int Int) Int)
(define-fun f ((x Int) (y Int)) Int (ite (<= x y) y x))

(declare-fun x () Int)
(declare-fun y () Int)

(assert(or (not(= (f x y) (f y x))) (not (and (<= x (f x y)) (<= y (f x y))))))

(check-sat)
(get-model)

enumerative, stochastic, solver guided, neural
bottom-up enumeration
observational equivalence - given the same input and spec, if the output is the same/incorrect, remove/prune - easy to do on some input
-> in synthesis by example - assumption is it will only work on that example - usually keep the smallest program - can order by cost instead - compositional
cegis - inductive synthesiser - something that can synthesise from examples
- verification oracle - either say good on all inputs or incorrect and return the inputs from which it is incorrect -> counterexample
- use a constraint solver for both
