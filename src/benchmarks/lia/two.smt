(set-logic LIA)

(declare-fun x () Int)

(declare-fun id1 (Int) Int)
(declare-fun id2 (Int) Int)
(declare-fun id3 (Int) Int)
(declare-fun id4 (Int) Int)

(assert
  (forall ((x Int))
    (= (id1 x) (id2 x) (id3 x) (id4 x) x)
  )
)

(check-sat)
(get-model)