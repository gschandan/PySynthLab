(set-logic LIA)

(synth-fun f ((a Int) (b Int)) Int)

(declare-var x Int)
(declare-var y Int)
(constraint (= (f x y) (f y x)))
(constraint (and (<= x (f x y)) (<= y (f x y))))