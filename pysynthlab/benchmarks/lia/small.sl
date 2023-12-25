(set-logic LIA)

(synth-fun f ((x Int) (y Int)) Int)

(declare-var x Int)
(declare-var y Int)
(constraint (= (f x y) (f y x)))
(constraint (and (<= x (f x y)) (<= y (f x y))))

for all x and y, 7 and 8 must be true

(check-synth)

;https://github.com/SyGuS-Org/benchmarks/blob/master/lib/CLIA_Track/from_2018/small.sl