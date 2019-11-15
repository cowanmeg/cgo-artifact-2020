#lang rosette

(require rosette/solver/smt/boolector
         rosette/solver/smt/z3)

(current-solver (boolector))
;(current-solver (z3 #:logic 'QF_BV))

; The bitwidth of Rosette's reasoning (bounds the number of registers, etc)
(current-bitwidth #f)

