(set-logic LIA)
(synth-fun max2 ((a Int) (b Int)) Int)

(declare-var x Int)
(declare-var y Int)

(constraint (>= (max2 x y) x))
(constraint (>= (max2 x y) y))
(constraint (or (= x (max2 x y)) (= y (max2 x y))))
(constraint (= (max2 x x) x))

(constraint (forall ((x Int) (y Int))
  (=> (>= x y) (= (max2 x y) x))))
(constraint (forall ((x Int) (y Int))
  (=> (>= y x) (= (max2 y x) y))))
(check-synth)
