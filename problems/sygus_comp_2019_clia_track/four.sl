(set-logic LIA)

(synth-fun id1 ((x Int)) Int (
    (Start Int (x (- Start) (+ Start x)))
))

(synth-fun id2 ((x Int)) Int (
    (Start Int (x (- Start) (+ Start x)))
))

(synth-fun id3 ((x Int)) Int (
    (Start Int (x (- Start) (+ Start x)))
))

(synth-fun id4 ((x Int)) Int (
    (Start Int (x (- Start) (+ Start x)))
))

(declare-var x Int)
(constraint (= (id1 x) (id2 x) (id3 x) (id4 x) x))

(check-synth)