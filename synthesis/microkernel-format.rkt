#lang rosette

(require rackunit
         rosette/solver/smt/boolector
         rosette/solver/smt/z3
         "compute-sketch.rkt"
         "reduce-sketch.rkt"
         "matrices.rkt"
         "arm-isa.rkt"
         "arm-printer.rkt"
         "print-extra.rkt")

(define fname "ukernel-intrin.c")
(define outport (open-output-file fname #:mode 'text #:exists 'replace))
(define time-fname "data/synthesis-time.csv")
(define timef (open-output-file time-fname #:mode 'text #:exists 'replace))
(fprintf timef "Config,Compute time (ms),Reduce time (ms),Verify time (ms), Total (ms)\n")

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

(define (search-format a-bw b-bw)
  (define m 8)
  (define k 8)
  (define n 1)
  (define bw 8)
  (define output-bw 16)
  
  (current-solver (boolector))
  (define compute-problem (make-object config% 1 k 1 a-bw b-bw 8 8 #t))
  (define compute-time-start (current-milliseconds))
  (define compute (search-compute-piecewise compute-problem #:min-instr 5 #:max-instr 10))
  (define compute-time (- (current-milliseconds) compute-time-start))

  (current-solver (z3 #:logic 'QF_BV))
  (define reduce-problem (make-object config-reduce% m k n a-bw b-bw bw output-bw #t))
  (define acc-reg (get-field out-reg reduce-problem))
  (define reduce-time-start (current-milliseconds))
  (define-values (reduce cost) (search-reduce-tree2 reduce-problem
                                                          #:min-depth 3
                                                          #:max-depth 5
                                                          #:start-cost 25))
  (define reduce-time (- (current-milliseconds) reduce-time-start))
  
  ; Verify each implementation
  (define verify-time-start (current-milliseconds))
  (check-true (unsat? (verify-prgm-last compute (send compute-problem get-initial-state)
                                        (send compute-problem get-intermediate-state) D)))
  (check-true (unsat? (verify-prgm-0 reduce
                                     (send reduce-problem get-intermediate-state2)
                                     (send reduce-problem get-final-state2)
                                     Q
                                     #:dst (reg acc-reg))))
  (define verify-time (- (current-milliseconds) verify-time-start))

  (printf "A~vW~v Compute time ~v ms Reduce time ~v ms Verify time ~v ms\n"
          b-bw a-bw compute-time reduce-time verify-time)
  (fprintf timef "A~vW~v,~v,~v,~v,~v\n"
          b-bw a-bw compute-time reduce-time verify-time (+ compute-time reduce-time verify-time))

  (stitch-together compute reduce m k n a-bw b-bw D Q acc-reg))

(print-imports #:outport outport)
(search-format 1 1)
(search-format 1 2)
(search-format 2 1)
(search-format 2 2)
(search-format 1 3)
(search-format 3 1)
(close-output-port outport)




  
         
