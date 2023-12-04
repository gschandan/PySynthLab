(set-logic LIA)

(declare-fun f (Int) Int)
(declare-fun g (Int Int) Int)

(declare-fun p () Int)
(declare-fun q () Int)

(assert (= (f p) (g p q)))

(check-sat)
(get-model)