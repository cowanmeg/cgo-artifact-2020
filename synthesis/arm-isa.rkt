#lang rosette

(provide (all-defined-out))
(require rosette/lib/match rosette/lib/angelic)

; The number of vector registers
(define NUM_REGISTERS 32)
(define REGISTER_BITS (inexact->exact (+ (log NUM_REGISTERS 2) 1)))

(define uint8 8)
(define uint16 16)

; Number of bits in a vector register
(define VREG_SIZE (* 16 uint8))
(define D_VREG_SIZE (/ VREG_SIZE 2))

; convenience to reference quad and double vector registers
; double register is either the top or bottom half of a quad register
; Q0 = [D1 D0]
(define Q #t)
(define D #f)

(define DTYPE_BITS 2)
(define 8bit (bv 0 DTYPE_BITS))
(define 16bit (bv 1 DTYPE_BITS))

; Vector instructions
(struct nop (rd)         #:transparent)
(struct vadd (q dtype rd ra rb)  #:transparent)
(struct vsub (q dtype rd ra rb)  #:transparent)
(struct vbic (q dtype rd ra rb)  #:transparent) ; bic(a,b) = a & (~b)
(struct vaddl (dtype rd ra rb)  #:transparent)
(struct vqadd (q dtype rd ra rb) #:transparent)
(struct vand (q dtype rd ra rb)  #:transparent)
(struct vcnt (q dtype rd ra)     #:transparent)
(struct vpadd (dtype rd ra rb)    #:transparent)
(struct vpaddl (q dtype rd ra)   #:transparent)
(struct vpadal (q dtype rd ra)   #:transparent)
(struct vlshf (q dtype rd ra imm)   #:transparent)
(struct vmov (rd ra) #:transparent)

; Interpret a program
(define (interpret prog state)
  (for/all ([prog prog])
    (for/fold ([state state])
              ([insn prog])
      (match insn
        [(nop rd) state]
        ; vector -> vector registers
        [(vadd q dtype rd ra rb)
         (let ([va (state-get-vreg state q ra)]
               [vb (state-get-vreg state q rb)])
           (for*/all ([dtype dtype] [va va] [vb vb])
             (state-set-vreg state q rd (vector-op bvadd dtype va vb))))]
        [(vsub q dtype rd ra rb)
         (let ([va (state-get-vreg state q ra)]
               [vb (state-get-vreg state q rb)])
           (for*/all ([dtype dtype] [va va] [vb vb])
             (state-set-vreg state q rd (vector-op bvsub dtype va vb))))]
        [(vbic q dtype rd ra rb)
         (let ([va (state-get-vreg state q ra)]
               [vb (state-get-vreg state q rb)])
           (for*/all ([dtype dtype])
             (state-set-vreg state q rd (bvand va (bvnot vb)))))]
        [(vaddl dtype rd ra rb)
         (let ([va (state-get-vreg state D ra)]
               [vb (state-get-vreg state D rb)])
           (for*/all ([dtype dtype] [va va] [vb vb])
             (state-set-vreg state Q rd (vector-promoting-add dtype va vb))))]
        [(vqadd q dtype rd ra rb)
         (let ([va (state-get-vreg state D ra)]
               [vb (state-get-vreg state D rb)])
           (for*/all ([dtype dtype])
             (state-set-vreg state Q rd (vector-op saturating-bvadd dtype va vb))))]
        [(vand q dtype rd ra rb)
         (let ([va (state-get-vreg state q ra)]
               [vb (state-get-vreg state q rb)])
           (for*/all ([va va] [vb vb])
             (state-set-vreg state q rd (bvand va vb))))]
        [(vcnt q dtype rd ra)
         (let ([va (state-get-vreg state q ra)])
           (for*/all ([va va])
             (state-set-vreg state q rd (vector-popcount va))))]
        [(vlshf q dtype rd ra imm)
         (let ([va (state-get-vreg state q ra)])
           (for*/all ([dtype dtype])
             (state-set-vreg state q rd (vector-left-shift va imm))))]
        [(vmov rd ra) ; Only moving D registers
         (let ([va (state-get-vreg state D ra)])
           (for*/all ([va va])
             (state-set-vreg state D rd va)))]
      
        ; Pairwise instructions
        [(vpadd dtype rd ra rb) ; Can only operate on D registers
         (let ([va (state-get-vreg state D ra)]
               [vb (state-get-vreg state D rb)])
           (for*/all ([dtype dtype] [va va] [vb vb])
             (state-set-vreg state D rd (pairwise-add dtype (concat vb va)))))]
        [(vpaddl q dtype rd ra)
         (let ([va (state-get-vreg state q ra)])
           (for*/all ([dtype dtype] [va va])
             (state-set-vreg state q rd (promoting-pairwise-add dtype va))))]
        [(vpadal q dtype rd ra)
         (let ([vd (state-get-vreg state q rd)]
               [va (state-get-vreg state q ra)])
           (for*/all ([dtype dtype] [vd vd] [va va])
             (state-set-vreg state q rd (promoting-pairwise-accumulate dtype vd va))))]))))

; Machine state
; An association list is a list of key-value pairs.
; We make it a struct just for convenience (we can call an assoclist
; as a procedure to retrieve a value).
(define (assoclist-get lst key)
  (let loop ([cases (assoclist-cases lst)])
      (cond [(null? cases) (assert #f)]
            [(equal? (caar cases) key) (cdar cases)]
            [else (loop (cdr cases))])))

; set always appends
(define (assoclist-set lst key val)
  (assoclist
   (cons (cons key val) (assoclist-cases lst))))

(define (assoclist-pop lst)
  (cdar (assoclist-cases lst)))

(struct assoclist (cases) #:transparent
  #:property prop:procedure assoclist-get)

; A machine state consists of vector registers
(struct state (vreg) #:transparent)

(define (state-get-vreg st q reg)
  (define q-reg (if q reg (d-to-q-reg reg)))
  (define lower (even? reg))
  (define contents (assoclist-get (state-vreg st) q-reg))
  (define out (cond
    [q      contents]
    [lower  (vreg-get-lower contents)]
    [else   (vreg-get-upper contents)]))
  out)

(define (state-get-last-vreg st)
  (assoclist-pop (state-vreg st)))

(define (state-set-vreg st q reg vals)
  (define q-reg (if q reg (d-to-q-reg reg)))
  (define lower (even? reg))
  (define new-vreg (cond
    [q      (assoclist-set (state-vreg st) q-reg vals)]
    [lower  (assoclist-set (state-vreg st) q-reg (concat (vreg-get-upper (assoclist-get (state-vreg st) q-reg)) vals))]
    [else   (assoclist-set (state-vreg st) q-reg (concat vals (vreg-get-lower (assoclist-get (state-vreg st) q-reg))))]))
  (state new-vreg))

(define (init-machine)
  (define vregs (build-list NUM_REGISTERS (lambda (x) (cons (reg x) (bv 0 VREG_SIZE)))))
  (state (assoclist vregs)))

(define (vreg-get-upper v)
    (extract (- VREG_SIZE 1) D_VREG_SIZE v))

(define (vreg-get-lower v)
    (extract (- D_VREG_SIZE 1) 0 v))
  
(define (odd? v)
  (bveq (extract 0 0 v) (bv 1 1)))

(define (even? v)
  (bveq (extract 0 0 v) (bv 0 1)))

(define (d-to-q-reg r)
  (for/all ([r r])
    (bvlshr r (reg 1))))

(define (reg x)
  (bv x REGISTER_BITS))

(define (bvdbl x)
  (bvshl x (reg 1)))

(define (bvhalf x)
  (bvlshr x (reg 1)))
    
(define (promote dtype)
  (bvadd dtype (bv 1 DTYPE_BITS)))

; Returns size of bitvector
(define (bitvector-size x)
  (assert (bv? x))
  ;(printf "~v\n" x)
  (match (type-of x)
    [(bitvector y) y]
    [_ (assert #f "Not a constant bitvector size")]))

; Popcount type - on ARM vcnt can only be preformed at 8-bit granularity
(define popcount-type? (bitvector 8))

; Simple popcount function. Returns number of set bits in a bitvector as a bitvecotr with same size.
(define (popcount x)
  (assert (popcount-type? x))
  (define sum-type (bitvector (bitvector-size x)))
  (apply bvadd (for/list ([i (bitvector-size x)])
                 (zero-extend (extract i i x) sum-type))))

(define (check-same-bitwidth x y)
  (assert (= (bitvector-size x) (bitvector-size y))))


(define (vector-combine x y)
  (check-same-bitwidth x y)
  (append x y))

(define (vector-op op dtype a b)
  ;(printf "vadd dtype: ~v  a:~v b:~v\n" dtype a b)
  ;(printf "vadd ~v: a:~v b:~v\n" dtype (bitvector-size a) (bitvector-size b))
  (for/all ([dtype dtype])
    (begin
      (define out
        (cond
          [(equal? dtype 8bit) (apply concat
                                      (for/list ([n (in-range (bitvector-size a) 0 -8)])
                                        (op (extract (- n 1) (- n 8) a) (extract (- n 1) (- n 8) b))))]
          [(equal? dtype 16bit) (apply concat
                                      (for/list ([n (in-range (bitvector-size a) 0 -16)])
                                        (op (extract (- n 1) (- n 16) a) (extract (- n 1) (- n 16) b))))]
          [else   (assert #t "vector-op on unsupported datatype")]))
      ;(printf "\t output: ~v\n" out)
      out)))

(define (vector-promoting-add dtype a b)
  ;(printf "vaddl dtype: ~v  a:~v b:~v\n" dtype a b)
  (for/all ([dtype dtype])
    (begin
      (define out
        (cond
          [(equal? dtype 8bit) (apply concat
                                      (for/list ([n (in-range (bitvector-size a) 0 -8)])
                                        (promoting-add 16 (extract (- n 1) (- n 8) a) (extract (- n 1) (- n 8) b))))]
          [else   (assert #t "vadd on unsupported datatype")]))
      ;(printf "\t output: ~v\n" out)
      out)))


(define (vector-popcount a)
  #;(printf "vcnt ~v\n" a)
  (define out
    (apply concat
           (for/list ([n (in-range (bitvector-size a) 0 -8)])
             (popcount (extract (- n 1) (- n 8) a)))))
  #;(printf "\t output: ~v\n" out)
  out)

(define (vector-left-shift a s)
  #;(printf "vcnt ~v\n" a)
  (define out
    (apply concat
           (for/list ([n (in-range (bitvector-size a) 0 -8)])
             (bvshl (extract (- n 1) (- n 8) a) s))))
  #;(printf "\t output: ~v\n" out)
  out)

; TODO: better way to detect overflow?
(define (saturating-bvadd a b)
  (let ([normal-add (bvadd a b)]
        [max-val (bv -1 (bitvector-size a))])
    (if (&& (bvuge normal-add a) (bvuge normal-add b)) normal-add max-val)))

; Adds bitvectors x and y and returns x+y with double the bitwidth
; (bitvector n) -> (bitvector n*2)
(define (promoting-add output-len x y)
  ;(check-same-bitwidth x y)
  (bvadd (sign-extend x (bitvector output-len)) (sign-extend y (bitvector output-len))))

; Given a vector registers [a b c d] returns [a+b c+d], where output bitvector length is double input
(define (promoting-pairwise-add dtype vreg)
  ;(printf "Promoting pairwise add dtype:~v ~v\n" dtype vreg)
  (for/all ([dtype dtype])
    (begin
      (define out
        (cond
          [(equal? dtype 16bit)
           (apply concat
                  (for/list ([n (in-range (bitvector-size vreg) 0 -32)])
                    (promoting-add 32 (extract (- n 1)(- n 16) vreg) (extract (- n 17) (- n 32) vreg))))]
          [(equal? dtype 8bit)
           (apply concat
                  (for/list ([n (in-range (bitvector-size vreg) 0 -16)])
                    (promoting-add 16 (extract (- n 1) (- n 8) vreg) (extract (- n 9) (- n 16) vreg))))]
          [else      (assert #t "Promoting pairwise add on unsupported datatype")]))
      ;(printf "\toutput ~v\n" out)
    out)))


; Given a vector registers [a b] [c d] returns [a+b c+d], where output bitvector length is same as input
(define (pairwise-add dtype vreg)
  (for/all ([dtype dtype])
    (begin
      ;(printf "Pairwise add dtype:~v input ~v\n" dtype vreg)
      (define out
        (cond
          [(equal? dtype 16bit)
           (apply concat
                  (for/list ([n (in-range (bitvector-size vreg) 0 -32)])
                    (bvadd (extract (- n 1)(- n 16) vreg) (extract (- n 17) (- n 32) vreg))))]
          [(equal? dtype 8bit)
           (apply concat
                  (for/list ([n (in-range (bitvector-size vreg) 0 -16)])
                    (bvadd (extract (- n 1) (- n 8) vreg) (extract (- n 9) (- n 16) vreg))))]
          [else      (assert #t "Pairwise add on unsupported datatype")]))
     ;(printf "\toutput ~v\n" out)
      out)))



; Given a vector register dst [a b] and src [c d e f] returns [a+c+d b+e+f]
(define (promoting-pairwise-accumulate dtype acc vreg)
  ;(printf "Promoting pairwise accumulate dtype:~v acc:~v reg:~v\n" dtype acc vreg)
  (for/all ([dtype dtype])
    (begin
      (define out
        (cond
          [(equal? dtype 16bit)
           (apply concat
                  (for/list ([n (in-range (bitvector-size vreg) 0 -32)])
                    (bvadd (extract (- n 1) (- n 32) acc)
                           (promoting-add 32 (extract (- n 1)(- n 16) vreg) (extract (- n 17) (- n 32) vreg)))))]
          [(equal? dtype 8bit)
           (apply concat
                  (for/list ([n (in-range (bitvector-size vreg) 0 -16)])
                    (bvadd (extract (- n 1) (- n 16) acc)
                           (promoting-add 16 (extract (- n 1) (- n 8) vreg) (extract (- n 9) (- n 16) vreg)))))]
          [else      (assert #t "Promoting pairwise acc on unsupported datatype")]))
      ;(printf "\toutput ~v\n" out)
    out)))





