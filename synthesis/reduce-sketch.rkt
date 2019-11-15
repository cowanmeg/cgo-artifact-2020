#lang rosette

(require "config.rkt"
         "arm-isa.rkt"
         "arm-printer.rkt"
         "matrices.rkt"
         "engine/synth.rkt")

(require rosette/lib/match rosette/lib/angelic
         racket/pretty rosette/lib/synthax
         (only-in rosette/base/core/bitvector bv-type)
         racket/trace)

(provide (all-defined-out))

(define (spec-0 final-state post-state q #:dst [dst (reg 0)])
  ;(printf "Correct      ~v\n" (state-get-vreg final-state Q (reg 0)))
  ;(printf "Written ~v\n" (state-get-vreg post-state Q dst))
  (if q
      (equal?  (state-get-vreg final-state Q (reg 0))
            (state-get-vreg post-state Q dst))
      (equal? (state-get-vreg final-state D (reg 0))
              (state-get-vreg post-state D dst))))

(define (spec-last final-state post-state q)
  ;(printf "Correct      ~v\n" (state-get-vreg final-state Q (reg 0)))
  ;(printf "Last written ~v\n" (state-get-last-vreg post-state))
  (if q
      (equal?  (state-get-vreg final-state Q (reg 0))
           (state-get-last-vreg post-state))
      (or (equal? (state-get-vreg final-state D (reg 0))
              (vreg-get-lower (state-get-last-vreg post-state)))
          (equal? (state-get-vreg final-state D (reg 0))
              (vreg-get-upper (state-get-last-vreg post-state))))))

; Verify output is in a specified register
(define (verify-prgm-0 prgm init-state final-state q #:dst [dst (reg 0)])
  (let ([post-state (interpret prgm init-state)])
    (verify (assert (spec-0 final-state post-state q #:dst dst)))))

; Verify output is in the last written register
(define (verify-prgm-last prgm init-state final-state q)
  (let ([post-state (interpret prgm init-state)])
    (verify (assert (spec-last final-state post-state q)))))

;;;;;;;;;;;;;;
; Choose number between [0, max)
(define (reg* max)
  (bounded-reg* (reg 0) max))

; Choose number between [min, max)
(define (bounded-reg* min max)
  ; Force to choose concrete values in the range
  (define-symbolic* r (bitvector REGISTER_BITS))
  (assert (bvult r max))
  (assert (bvuge r min))
  r) 
;;;;;;;;;;;;;;;

;;;;;;;;;;;;;;;;;
; Reduction sketch
(struct level (instr outputs outdtype) #:transparent)

(define (instr x)
  (bv x 6))

; Forces reduction to follow a tree pattern
(define (tree-reduce d-inputs depth dtype #:dst [dst (reg 0)])
   (let loop ([i 1] [inputs (reg d-inputs)] [dtype dtype])
     (define end-reg-d d-inputs)
     (define end-reg-q (/ d-inputs 2))
     (define tree
       (cond
         ((= i depth)
           ; The last level of the tree needs to accumulate into the output
          (choose*
           (level
             (for/list ([i (in-range 0 end-reg-q)])
               (letrec ([rd (bvadd dst (reg i))]
                        [ra (reg i)])
                 (vpadal Q dtype rd ra)))
             inputs (promote dtype))
           
           (level
            (for/list ([i (in-range 0 end-reg-q)])
              (letrec ([rd (bvadd dst (reg i))]
                       [ra (reg i)])
                (vadd Q dtype rd ra rd)))
            (bvhalf inputs) dtype)))
    
       (else
        ; Inner levels
          (choose*
           (level
              (for/list ([i (in-range end-reg-d)])
                (letrec ([rd (reg i)]
                         [ra (bvdbl (reg i))])
                  (vpaddl D dtype rd ra)))
              inputs (promote dtype))
           
           (level
            (for/list ([i (in-range 0 end-reg-d 2)])
              (letrec ([rd (bvhalf (reg i))]
                       [ra (reg i)]
                       [rb (bvadd ra (reg 1))])
                (choose*
                 (vpadd dtype rd ra rb)
                 #;(vadd D dtype rd ra rb))))
            (bvhalf inputs) dtype)

            (level
            (for/list ([i (in-range end-reg-q)])
              (letrec ([rd (reg i)]
                       [ra (reg i)])
                (vpaddl Q dtype rd ra)))
            inputs (promote dtype))
           ))))

     (define instrs (level-instr tree))
     (define num-outputs (level-outputs tree)) 
     (define cost num-outputs) ; Each output required 1 instruction, all equal cost
     (define next-dtype (level-outdtype tree))
     
     (if (< i depth)
         (let-values ([(next-instrs next-cost) (loop (+ i 1) num-outputs next-dtype)])
           (values (append instrs next-instrs)
                   (bvadd cost next-cost)))

         (values instrs cost))))

; Gets rid of the extra instructions created from the rectangle tree and make it a real tree
(define (filter-instr prgm d-inputs)
  (define inputs d-inputs)
  (define (keep-instr? instr)
    (match instr
      [(vadd q dtype rd ra rb)
       (define keep (if q
           (or (bveq ra (reg 0)) (bvult rb (bvhalf inputs)))
           (or (bvult rb inputs))))
       ; After last instruction of the level - reduce the number of inputs by half
       (when (bveq rb (bvsub inputs (reg 1))) (set! inputs (bvhalf inputs)))
       keep]
      [(vpadd dtype rd ra rb)
       (let ([keep (bvult rb inputs)])
         (when (bveq rb (bvsub inputs (reg 1))) (set! inputs (bvhalf inputs)))
         keep)]
      [(vpaddl q dtype rd ra)
       (if q
           (or (bveq rd (reg 0)) (bvult rd (bvhalf inputs)))
           (bvult rd inputs))]
      [(vpadal q dtype rd ra)
       (if q
           (or (bveq rd (reg 0)) (bvult rd (bvhalf inputs)) (bveq rd (reg 16)))
           (bvult rd inputs))]
      [else (assert #t "Unknown instruction in filter")]))
    
  (filter keep-instr? prgm))

;;;;;;

(define (search-reduce-tree2 problem
                       #:min-depth   [min-depth 1]
                       #:max-depth   [max-depth 5]
                       #:start-cost  [start-cost 25]
                       #:desired-cost [desired-cost 0])

  (let ([maxmb2 (send problem get-maxmb2)]
        [acc (send problem get-acc)]
        [init-state (send problem get-intermediate-state2)]
        [final-state (send problem get-final-state2)]
       
        [n (send problem num-reduce-vectors)]
        [q (if (= (* (get-field OUTPUT_BW problem) (get-field M problem)) VREG_SIZE) Q D)]
        [out-reg (reg (get-field out-reg problem))]
        [start-cost (reg start-cost)]
        [desired-cost (reg desired-cost)])
    (let loop ([i min-depth])
      ;(printf "Tree depth = ~v\n" i)
      (define-values (sketch sketch-cost) (tree-reduce n i 8bit #:dst out-reg))
      (define post-state (interpret sketch init-state))
      (define-values (first-sol next)
            (synth #:forall (list maxmb2 acc)
                        #:guarantee (assert (&& (spec-0 final-state post-state q #:dst out-reg)
                                                (bvult sketch-cost start-cost)))
                        #:guess-first-cex #t))

      (cond
        [(sat? first-sol) ; Solution found at that depth - keep lowering the cost until instructions are minimized
         (let ([first-cost (evaluate sketch-cost (complete-solution first-sol (symbolics sketch-cost)))])
           (let cost-loop ([cost first-cost] [sol first-sol])
             (define next-sol (next (bvult sketch-cost cost)))
             (if (unsat? next-sol) 
                 (begin
                  (values (filter-instr (evaluate sketch (complete-solution sol (symbolics sketch))) (reg n)) (bitvector->natural cost)))
                 (begin 
                   (define next-cost (evaluate sketch-cost (complete-solution next-sol (symbolics sketch-cost))))
                   (cost-loop next-cost next-sol)))))]
        
        [(< i max-depth) (loop (+ i 1))]
        [else (values unsat -1)]))))

             




