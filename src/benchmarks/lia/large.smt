;(set-logic LIA)
;
;(declare-fun x  () Int)
;(declare-fun y () Int)
;
;(define-fun func ((x Int)) Int
;            (+ (* x 100) 1000))
;
;(define-fun f ((x Int) (y Int)) Int
;    (ite (<= x y)
;            (+ (* x 100) 1000)
;            (+ (* y 100) 1000)
;            ))
;
;(assert (or (not (= (f x y) (f y x)))
; (not (and 
;    (>= (func x) (f x y)) 
;    (>= (func y) (f x y))))))
;(check-sat)

(declare-fun y () Int)
(declare-fun x () Int)
(declare-fun func (Int) Int)
(assert
 (let ((?x32 (* 100 y)))
 (let ((?x33 (+ ?x32 1000)))
 (let ((?x30 (* 100 x)))
 (let ((?x31 (+ ?x30 1000)))
 (let ((?x34 (ite (<= x y) ?x31 ?x33)))
 (let ((?x14 (func y)))
 (or (not (= ?x34 (ite (<= y x) ?x33 ?x31))) (not (and (>= (func x) ?x34) (>= ?x14 ?x34)))))))))))
(check-sat)

need to also substitute defined functions
seed solver differently
or add asserts that x and y can't equal incorrect inputs
