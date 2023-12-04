(set-logic LIA)

(synth-fun f ((x Int)) Int)
(synth-fun g ((a Int) (b Int)) Int)

(declare-var p Int)
(declare-var q Int)

(constraint (= (f p) (g p q)))
(check-synth)