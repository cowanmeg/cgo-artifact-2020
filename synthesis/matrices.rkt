#lang rosette

(require racket/class
         "arm-isa.rkt")
(provide (all-defined-out))


; combine list of bitvectors into one single one
(define (bv-combine lst)
  (apply concat (reverse lst)))

; pads a vreg to x bits with 0s in the front
(define (pad vreg bw)
  (if (< (bitvector-size vreg) bw)
      (zero-extend vreg (bitvector bw))
      vreg))

; broadcast the val into a bitvector of size
(define (broadcast-val val size)
  (let ([repeats (/ size (bitvector-size val))])
    (apply concat (for/list ([i (in-range repeats)]) val))))

; decide wether using D or Q registers
; D Regitser configurations int4x8 int8x4 int16x2
; Q Register configurations int4x16 int8x8 int16x4
(define (vector-type k BW)
  (cond
    [(= (* BW k) D_VREG_SIZE) D]
    [(= (* BW k) VREG_SIZE) Q]
    [else (begin (displayln "Problem doesn't fit machine vector length") D)]))

; Define symbolic or concrete matrix
; dimensions axbxc with each element a bitvector of size bw
(define (create-matrix a b c bw sym zero-bits)
  (for/list ([i a])
    (for/list ([j b])
      (for/list ([k c])
        (cond
          [(and sym (= zero-bits 0))
           (begin (define-symbolic* v (bitvector bw)) v)]
          [(and sym (> zero-bits 0))
           (begin
             (define symbolic-bits (- bw zero-bits))
             (define-symbolic* sym-v (bitvector symbolic-bits))
             (define concrete-v (integer->bitvector (random (expt 2 zero-bits)) (bitvector zero-bits)))
             (concat sym-v concrete-v))]
          [else
           (integer->bitvector (random (expt 2 bw)) (bitvector bw))]
          )))))


; MxN NxK matrix multiply
; K is vectorized axis
(define config%
  (class object%
    (super-new)
    (init-field
     [M 4]
     [K 8]
     [N 1]
     [A_BW 1]
     [B_BW 2]
     [BW 8]
     [OUTPUT_BW 16]
     [SYM #t] ;Use symbolic values
     [ZERO-BITS 0] ; if sym is true, number of concrete bits
     )

    ; Getter methods - matrices
    (define/public (get-ma) ma)
    (define/public (get-mb) mb)
    (define/public (get-maxmb) maxmb)
    (define/public (get-ma*mb) ma*mb)

    ; state
    (define/public (get-initial-state) (load-initial-state ma mb))
    (define/public (get-intermediate-state) (load-initial-state-reduce maxmb))
    (define/public (get-int-bw-state a_bw b_bw) (load-initial-state-reduce (maxmb-bitplane a_bw b_bw)))
    (define/public (get-final-state) (load-final-state ma*mb))
    (define/public (num-outputs) (* M N))

    (define ma (create-matrix M A_BW K BW SYM ZERO-BITS))
    (define mb (create-matrix N B_BW K BW SYM ZERO-BITS))

    (define ma*mb
      (for/list ([i M])
        (for/list ([j N])
          (foldl (lambda (indices acc)
                   (let ([k (first indices)] [a_bw (second indices)] [b_bw (third indices)])
                     (bvadd (sign-extend (bvshl (bvsub
                                                 (popcount (bvand
                                                            (list-ref (list-ref (list-ref ma i) a_bw) k)
                                                            (list-ref (list-ref (list-ref mb j) b_bw) k)))
                                                 (popcount (bvand
                                                            (list-ref (list-ref (list-ref ma i) a_bw) k)
                                                            (bvnot (list-ref (list-ref (list-ref mb j) b_bw) k)))))
                                                (bvadd (bv a_bw BW)(bv b_bw BW))) (bitvector OUTPUT_BW))
                            acc)))
                 (bv 0 (bitvector OUTPUT_BW))
                 (for*/list ([k (range K)] [a_bw (range A_BW)] [b_bw (range B_BW)]) (list k a_bw b_bw))))))

    ; intermediate - K is the vectorized axis
    (define maxmb (for*/list ([i M] [j N])
                    (for/list ([k K])
                             (foldl (lambda (indices acc)
                                      (let ([a_bw (first indices)] [b_bw (second indices)])
                                        (bvadd (bvshl
                                                (bvsub
                                                 (popcount (bvand
                                                            (list-ref (list-ref (list-ref ma i) a_bw) k)
                                                            (list-ref (list-ref (list-ref mb j) b_bw) k)))
                                                 (popcount (bvand
                                                            (bvnot (list-ref (list-ref (list-ref ma i) a_bw) k))
                                                            (list-ref (list-ref (list-ref mb j) b_bw) k))))
                                                      (bvadd (bv a_bw BW)(bv b_bw BW)))
                                               acc)))
                                    (bv 0 (bitvector BW))
                                    (for*/list ([a_bw (range A_BW)] [b_bw (range B_BW)]) (list a_bw b_bw))))))

    ; Generate an intermediate example
    (define (maxmb-bitplane a_bw b_bw)
      ;(printf "starting to generate bitplane\n")
      (define out (for*/list ([i M] [j N])
                    (for/list ([k K])
                             (bvshl
                              (bvsub
                               (popcount (bvand
                                          (list-ref (list-ref (list-ref ma i) a_bw) k)
                                          (list-ref (list-ref (list-ref mb j) b_bw) k)))
                               (popcount (bvand
                                          (bvnot (list-ref (list-ref (list-ref ma i) a_bw) k))
                                          (list-ref (list-ref (list-ref mb j) b_bw) k))))
                              (bvadd (bv a_bw BW)(bv b_bw BW))))))
      ;(printf "int ~v \n" out)
      out)
   
   
    ; Load the matrices ma and mb^T into vector registers both in row major.
    (define (load-initial-state ma mb)
      (let ([m (length ma)] [a_bw (length (first ma))] [k (length (first (first ma)))]
                            [n (length mb)] [b_bw (length (first mb))])                    
        (unless (<= (+ (* m a_bw) (* n b_bw)) NUM_REGISTERS)
          (error 'load-initial-state "not enough regs"))
        (define st (init-machine))
        (define q (vector-type K BW))
        (define ma-st (for*/fold ([st st])
                   ([x (in-range m)] [y (in-range a_bw)])
          (let ([vals (list-ref (list-ref ma x) y)] [i (+ y (* x a_bw))])
            ;(printf "Load init state ~v idx ~v ~v\n" (if q "Q" "D") i vals)
            (state-set-vreg st q (reg i) (pad (bv-combine vals) D_VREG_SIZE)))))
        (for*/fold ([ma-st ma-st])
                   ([x (in-range n)] [y (in-range b_bw)])
          (let ([vals (list-ref (list-ref mb x) y)] [i (+ y (* x b_bw) (* m a_bw))])
            ;(printf "Load init state ~v idx ~v ~v\n" (if q "Q" "D") i vals)
            (state-set-vreg ma-st q (reg i) (pad (bv-combine vals) D_VREG_SIZE))))))

    (define (load-initial-state-reduce m)
      (unless (<= (length m) NUM_REGISTERS)
          (error 'load-initial-state "not enough regs"))
      ;(printf "maxmb ~v \n" (first maxmb))
      (define q (vector-type (length (first m)) BW))
      (define st (init-machine))
      (for/fold ([st st])
                ([i (in-naturals)] [val (in-list m)])
        ;(printf "Load int state ~v idx ~v ~v\n" (if q "Q" "D") i val)
        (state-set-vreg st q (reg i) (pad (bv-combine val) D_VREG_SIZE))))


    (define (load-final-state m)
      ;(printf "ma*mb ~v\n" ma*mb)
      (unless (<= (length m) NUM_REGISTERS)
        (error 'load-final-state "not enough regs"))
      (define q (vector-type M OUTPUT_BW))
      (define st (init-machine))
      ;(printf "Load final state ~v idx ~v\n" (if q "Q" "D") 0)
      (state-set-vreg st q (reg 0) (pad (bv-combine (flatten m)) D_VREG_SIZE)))
         
    ; Range functions
    (define/public (num-input-vectors)
      (let ([a-vectors (* M A_BW)]
            [b-vectors (* N B_BW)])
        (+ a-vectors b-vectors)))

    ; Number of D registers taken up by input
    (define/public (num-reduce-vectors)
      ; if input vectors fit in Q registers, multiply by 2
      (let ([q-multiplier (if (= (* K BW) VREG_SIZE) 2 1)])
        (* M N q-multiplier)))

    
    ;[start, end] of D registers that matrix a takes up
    (define/public (range-a-vectors)
      (let ([q-multiplier (if (= (* K BW) VREG_SIZE) 2 1)])
        (values (reg 0) (reg (* M A_BW q-multiplier)))))

    ; [start, end] of D registers that matrix b takes up
    (define/public (range-b-vectors)
      (letrec ([q-multiplier (if (= (* K BW) VREG_SIZE) 2 1)]
               [a-vectors (* M A_BW q-multiplier)]
               [b-vectors (* N B_BW q-multiplier)])
        (values  (reg a-vectors) (reg (+ a-vectors b-vectors)))))

    (define/public (pretty-print)
      (printf "A: ~vx~v ~v-bits B: ~vx~v ~v-bits\n" M K A_BW K N B_BW)
      (printf "Packed into ~v-bits, outputs acculumated in ~v-bits\n" BW OUTPUT_BW)
      (printf "Number of symbolic bits: ~v. Number of hardcoded bits: ~v\n" SYM ZERO-BITS))))

;;;;;;;;
; MxN NxK matrix multiply
; K is vectorized axis
; Only models from the intermediate step 
(define config-reduce%
  (class object%
    (super-new)
    (init-field
     [M 4]
     [K 8]
     [N 1]
     [A_BW 1]
     [B_BW 2]
     [BW 8]
     [OUTPUT_BW 16]
     [SYM #t] ;Use symbolic values
     )

    ; Getter methods - matrices
    (define/public (get-maxmb2) maxmb2)
    (define/public (get-ma*mb2) ma*mb2)
    (define/public (get-acc) accumulate)
    (define/public (get-intermediate-state2) (load-initial-state-reduce maxmb2))
    (define/public (get-final-state2) (load-final-state ma*mb2))
  
    (define/public (num-outputs) (* M N))

    (field (out-reg 16))

    (define maximum
      ; For 1-bit/1-bit the max and min values ares 7 and -7
      (let ([bw-combos
        (for*/list ([a_bw (range A_BW)] [b_bw (range B_BW)])
          (cons a_bw b_bw))])
        (foldl (lambda (ab_bw val)
                 (+ val (* 7 (expt 2 (+ (car ab_bw) (cdr ab_bw))))))
               0
               bw-combos)))
    
    ; Intermediate starting point. Instead of computing from symbolic inputs, bound all elements
    ; based off the bitwidth
    (define maxmb2
      (begin
        (let ([max-value (integer->bitvector maximum (bitvector BW))]
              [min-value (integer->bitvector (- 0 maximum) (bitvector BW))]
              [number-vectors (* M N)]
              [number-elements K])
        (for/list ([m number-vectors])
          (for/list ([e (in-range number-elements)])
                     (if SYM
                         (begin
                           (define-symbolic* v (bitvector BW))
                           (assert (bvsle v max-value))
                           (assert (bvsge v min-value))
                           v)
                         (integer->bitvector (random (- 0 maximum) (+ 1 maximum)) (bitvector BW))))))))

    (define accumulate
      (for/list ([m (in-range M)])
        (if SYM
            (begin
              (define-symbolic* acc (bitvector OUTPUT_BW))
              (assert (bvsle acc (integer->bitvector 150 (bitvector OUTPUT_BW))))
              (assert (bvsge acc (integer->bitvector -150 (bitvector OUTPUT_BW))))
              acc)
            (bv 10 OUTPUT_BW))))
    
    (define ma*mb2
      ; accumulate result in a non-zero output - bound so it doesn't overflow
      (for/list ([m maxmb2] [acc accumulate])
        (bvadd acc
               (apply bvadd
                      (for/list ([n m])
                        (sign-extend n (bitvector OUTPUT_BW)))))))
    
  
    (define (load-initial-state-reduce m)
      (unless (<= (length m) NUM_REGISTERS)
          (error 'load-initial-state "not enough regs"))
      ;(printf "maxmb2 ~v\n" (first maxmb2))
      (define q (vector-type (length (first m)) BW))
      (define st (init-machine))
      (define in-st (for/fold ([st st])
                ([i (in-naturals)] [val (in-list m)])
        ;(printf "Load int state ~v idx ~v ~v\n" (if q "Q" "D") i val)
        (state-set-vreg st q (reg i) (pad (bv-combine val) D_VREG_SIZE))))
      
      ;The final output will be accumulated in the destination register
      (state-set-vreg in-st Q (reg out-reg) (bv-combine accumulate)))

    (define (load-final-state m)
      ;(printf "ma*mb2 ~v\n" ma*mb2)
      (unless (<= (length m) NUM_REGISTERS)
        (error 'load-final-state "not enough regs"))
      (define q (vector-type M OUTPUT_BW))
      (define st (init-machine))
      ;(printf "Load final state ~v idx ~v\n" (if q "Q" "D") 0)
      (state-set-vreg st q (reg 0) (pad (bv-combine (flatten m)) D_VREG_SIZE)))
         
    ; Range functions
    (define/public (num-input-vectors)
      (let ([a-vectors (* M A_BW)]
            [b-vectors (* N B_BW)])
        (+ a-vectors b-vectors)))

    ; Number of D registers taken up by input
    (define/public (num-reduce-vectors)
      ; if input vectors fit in Q registers, multiply by 2
      (let ([q-multiplier (if (= (* K BW) VREG_SIZE) 2 1)])
        (* M N q-multiplier)))

    
    ;[start, end] of D registers that matrix a takes up
    (define/public (range-a-vectors)
      (let ([q-multiplier (if (= (* K BW) VREG_SIZE) 2 1)])
        (values (reg 0) (reg (* M A_BW q-multiplier)))))

    ; [start, end] of D registers that matrix b takes up
    (define/public (range-b-vectors)
      (letrec ([q-multiplier (if (= (* K BW) VREG_SIZE) 2 1)]
               [a-vectors (* M A_BW q-multiplier)]
               [b-vectors (* N B_BW q-multiplier)])
        (values  (reg a-vectors) (reg (+ a-vectors b-vectors)))))

    (define/public (pretty-print)
      (printf "A: ~vx~v ~v-bits B: ~vx~v ~v-bits\n" M K A_BW K N B_BW)
      (printf "Packed into ~v-bits, outputs acculumated in ~v-bits\n" BW OUTPUT_BW)
      (printf "Inputs symbolic ~v\n" SYM))))