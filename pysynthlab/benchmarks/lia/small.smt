(set-logic LIA)

;(declare-fun f (Int Int) Int)
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
