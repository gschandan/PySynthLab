(set-logic LIA)

(declare-fun f (Int Int) Int)

(declare-fun x () Int)
(declare-fun y () Int)

(assert (= (f x y) (f y x)))
(assert (and (<= x (f x y)) (<= y (f x y))))

(check-sat)
(get-model)