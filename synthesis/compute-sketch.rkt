#lang rosette
; Compute sketch

(require "config.rkt"
         "arm-isa.rkt"
         "arm-printer.rkt"
         "reduce-sketch.rkt"
         "matrices.rkt"
         "engine/synth.rkt")

(require rosette/lib/match rosette/lib/angelic
         racket/pretty rosette/lib/synthax
         (only-in rosette/base/core/bitvector  bv-type)
         racket/trace)

(provide search-compute-piecewise)

; Generates (bv x 8) where x is from [1, upper]
(define (choose-range* upper)
  (apply choose* (for/list ([i (+ 1 upper)])
                   (bv i 8))))

; Compute sketch 1: Generic, no ordering of instruction enforced
(define (compute-inst*/bounded i a-min a-max b-min b-max dst-reg q bw-max)
  (define dtype 8bit)
  (letrec ([rd dst-reg] ; next open register
           [ra (reg* (bvadd b-max i))]
           [rb (bounded-reg* ra (bvadd b-max i))]
           [imm (choose-range* bw-max)])
    (choose*
       (vand q dtype rd ra rb)
       (vbic q dtype rd rb ra)
       (vlshf q dtype rd ra imm)
       (vcnt q dtype rd ra)
       (vadd q dtype rd ra rb)
       (vsub q dtype rd ra rb)
       #;(vmov rd ra))))

(define (compute-cost prgm)
 (foldl (lambda (instr cost)
  (match instr
    ; Prefer bitwise operations
    [(vand q dtype rd ra rb) (bvadd cost (reg 1))]
    [(vbic q dtype rd ra rb) (bvadd cost (reg 1))]
    [ _                      (bvadd cost (reg 2))]))
  (reg 0)
  prgm))
       
(define (sketch-n-instr num-instr problem final-reg q bw-max)
  (let-values ([(a-min a-max) (send problem range-a-vectors)]
               [(b-min b-max) (send problem range-b-vectors)])
    
    (for/list ([i (in-range num-instr)])
      (define dst-reg (if (= i (- num-instr 1)) final-reg (bvadd (reg i) b-max)))
      (compute-inst*/bounded (reg i) a-min a-max b-min b-max dst-reg q bw-max))))


; Synthesize "multiplication" between each bitplane and then add them
(define (search-compute-piecewise problem
                        #:min-instr [min-instr 2]
                        #:max-instr [max-instr 10])
  (let ([ma (send problem get-ma)]
        [mb (send problem get-mb)]
        [init-state (send problem get-initial-state)]
        [q (if (= (* (get-field OUTPUT_BW problem) (get-field K problem)) VREG_SIZE) Q D)]
        [A_BW (get-field A_BW problem)]
        [B_BW (get-field B_BW problem)]
        [dst 16]
        [start-cost (reg 12)])
    (define int-dst dst)
    (define parts (apply append
            (for*/list ([a_bw (in-range A_BW)][b_bw (in-range (get-field B_BW problem))])
              (let ([final-state (send problem get-int-bw-state a_bw b_bw)]
                    [out-reg (reg int-dst)])
                (define bw-max (+ a_bw b_bw))
                (set! int-dst (+ 1 int-dst))
                (let loop ([i min-instr])
                  ;(printf "Instructions = ~v\n" i)
                  (define sketch (sketch-n-instr i problem out-reg q bw-max))
                  (define cost (compute-cost sketch))
                  (define post-state (interpret sketch init-state))
                  (define-values (first-sol next)
                   (synth #:forall (list ma mb)
                          #:guarantee (assert (&& (spec-last final-state post-state q) (bvult cost start-cost)))
                          #:guess-first-cex #t))
                 (cond
                   [(sat? first-sol) (begin
                                 (define prgm (evaluate sketch (complete-solution first-sol (symbolics sketch))))
                                 (define first-cost (compute-cost prgm))
                                 (let cost-loop ([cost first-cost] [sol first-sol])
                                   (define prgm (evaluate sketch (complete-solution sol (symbolics sketch))))
                                   (define current-cost (compute-cost prgm))
                                   (define next-sol (next (bvult (compute-cost sketch) cost)))
                                   (if (unsat? next-sol) 
                                       (begin
                                         (define prgm-part (evaluate sketch (complete-solution sol (symbolics sketch))))
                                         (set! min-instr (length prgm-part))
                                         prgm-part)
                                       (begin
                                         (define next-prgm (evaluate sketch (complete-solution next-sol (symbolics sketch))))
                                         (define next-cost  (compute-cost next-prgm))
                                         (cost-loop next-cost next-sol)))))
                                 
                                 ]
                   [(< i max-instr) (loop (+ i 1))]
                   [else unsat]))))))
    (define combine
      (for/list ([i (in-range 1 (* A_BW B_BW))])
        (vadd q 8bit (reg dst) (reg dst) (bvadd (reg i) (reg dst)))))
    (append parts combine)))
