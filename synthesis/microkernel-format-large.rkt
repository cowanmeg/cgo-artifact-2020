#lang rosette

(require rackunit
         rosette/solver/smt/boolector
         rosette/solver/smt/z3
         "reduce-sketch.rkt"
         "compute-sketch.rkt"
         "matrices.rkt"
         "arm-isa.rkt"
         "arm-printer.rkt"
         "print-extra.rkt")

(define fname "ukernel-intrin-large.c")
(define outport (open-output-file fname #:mode 'text #:exists 'replace))

; Format everything together
(define (stitch-together compute reduce m k n a-bw b-bw input-type output-type acc-reg)
  (define assembly #f)
  (print-header k a-bw b-bw #:outport outport)

  ; Compute prgm needs to be applied m times
  ; Assumes that n = 1
  (define output-vars
    (for*/list ([i (in-range m)] [j (in-range n)])
      
      (define ma-inputs
        (for/list ([bw (in-range a-bw)] [idx (in-naturals)])
          (name-map input-type (reg idx) (string-join (list "a" (number->string bw) "[" (number->string i) "]") ""))))
      (define ma-regs (length ma-inputs))
      (define mb-inputs
        (for/list ([bw (in-range b-bw)] [idx (in-naturals)])
          (name-map input-type (reg (+ idx ma-regs)) (string-join (list "b" (number->string bw)) ""))))
      (define inputs (append ma-inputs mb-inputs))
      
      (define prefix (string-join (list "x" (number->string i) (number->string j)) ""))
      (define output (name-map input-type (reg i) (string-join (list "y" (number->string i)) "")))
      (print-prgm compute #:inputs inputs #:output output #:prefix prefix #:assembly assembly #:outport outport)
      output))

  ; Reduce portion
  (define reduce-inputs
    (append output-vars (list (name-map output-type (reg acc-reg) "acc"))))
  (define reduce-output (name-map output-type null "out"))
  (print-prgm reduce #:inputs reduce-inputs #:output reduce-output #:assembly assembly #:outport outport)

  (print-ending #:outport outport))


(define (search-format a-bw b-bw #:min-depth [min-depth 4])
  (define m 8)
  (define k 16)
  (define n 1)
  (define bw 8)
  (define output-bw 16)

  ; boolector faster for compute sketch
  (current-solver (boolector))
  (define compute-problem (make-object config% 1 k 1 a-bw b-bw 8 8 #t))
  (define compute (search-compute-piecewise compute-problem #:min-instr 5 #:max-instr 10))

  ; z3 faster for reduce sketch
  (current-solver (z3  #:logic 'QF_BV))
  (define reduce-problem (make-object config-reduce% m k n a-bw b-bw bw output-bw #t))
  (define acc-reg (get-field out-reg reduce-problem))
  (define-values (reduce cost) (search-reduce-tree2 reduce-problem
                                                          #:min-depth min-depth
                                                          #:max-depth 7
                                                          #:start-cost 25))

  (stitch-together compute reduce m k n a-bw b-bw Q Q acc-reg))

(print-imports #:outport outport)
(search-format 1 1)
(search-format 1 2 #:min-depth 5)
(search-format 2 2 #:min-depth 5)
(close-output-port outport)
  
         
