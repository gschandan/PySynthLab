(set-logic LIA)

(synth-fun f1 ((x Int)) Int
  ((y Int) (z Int))
  ((+ x y) (- x z)))

(synth-fun f2 ((a Int) (b Int)) Int
  ((c Int) (d Int))
  ((+ a c) (- b d)))

(declare-var p Int)
(declare-var q Int)

(constraint (= (f1 p) (f2 p q)))
(check-synth)