(set-logic LIA)
(synth-fun f ((a Int) (b Int)) Int)
(define-fun func ((x Int)) Int
            (+ (* x 100) 1000))
(declare-var x Int)
(declare-var y Int)
(constraint (= (f x y) (f y x)))
(constraint (and (>= (func x) (f x y)) (>= (func y) (f x y))))
(check-synth)