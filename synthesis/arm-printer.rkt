#lang rosette

(require "arm-isa.rkt")

(provide print-prgm prgm-len cleanup-prgm name-map)

; Program length - ignore nops
(define (prgm-len prgm)
  (if (list? prgm)
      (foldl (lambda (x acc) (if (nop? x) acc (+ 1 acc))) 0 prgm)
      0))

; Removes nops from list of instructions
(define (cleanup-prgm prgm)
  (if (list? prgm)
      (filter (lambda (x) (not (nop? x))) prgm)
      '()))

(struct var (type id) #:transparent)
(struct name-map (type register name) #:transparent)

; Prints assembly instructions
(define (print-prgm prgm
                    #:inputs [inputs null]
                    #:output [output null]
                    #:prefix [prefix "x"]
                    #:assembly [assembly #f]
                    #:outport [outport (current-output-port)])
 
  ; ex. uint8x8_t d0
  ; updates output registers variable mapping by appending an _
  (define (format-intrin-output i register->name r q)
    ; Outputs might use predefined nmaes from output - check if this is the last intr first
    (if (and (not (null? output)) (= (- (length prgm) 1) i))
        (name-map-name output)
        (begin
          (define d-reg (if q (bvdbl r) r))
          (define post-fix (if (odd? d-reg) "o" "_"))
          (define old-id
                          
            (if (hash-has-key? register->name d-reg)
             (let ([v (hash-ref register->name d-reg)])
               (var-id v))
 
             (let ([new-id (string-append prefix (number->string (bitvector->natural d-reg)))])
               (hash-set! register->name reg (var q new-id))
               new-id)))
        
          ;(define old-id (format-intrin-input register->name d-reg D))
          (define output-id (string-append old-id post-fix))
          (hash-set! register->name d-reg (var q output-id))
          ; If assigning a Q register also assign the neighbor
          (when q
            (hash-set! register->name (bvadd (reg 1) d-reg) (var Q output-id)))
          output-id)))

  ; whole intrin op
  ; operands are registers - supports 1 or 2
  ; Ex. uint8x8t d0_ = vpadd_u8(d0, d1) or vpaddl_u8(d0)
  (define (format-intrin op i register->name qs dtype output-dtype signed . regs)
    (define operand-list (for/list ([reg (cdr regs)] [q (cdr qs)])
                           (format-intrin-input register->name reg q signed dtype)))
    (define output-id (format-intrin-output i register->name (first regs) (first qs)))
    (define output-str (format-neon-output output-id dtype signed (first qs)))
    (fprintf outport "\t")
    (fprintf outport output-str)
    (fprintf outport (string-join operand-list
                         ", "
                         #:before-first (format-neon-op op dtype signed (first qs))
                         #:after-last ");\n")))

  ; whole intrin op
  ; first operand is a register second is an imm
  ; Ex. uint8x8t d0_ = vlshf_u8(d0, 2)
  (define (format-intrin-imm op i register->name qs dtype output-dtype signed dst-reg src-reg imm)
    (define operand-list (list (format-intrin-input register->name src-reg (second qs) signed dtype) (number->string (bitvector->natural imm))))
    (define output-id (format-intrin-output i register->name dst-reg (first qs)))
    (define output-str (format-neon-output output-id output-dtype signed (first qs)))
    (fprintf outport "\t")
    (fprintf outport output-str)
    (fprintf outport (string-join operand-list
                         ", "
                         #:before-first (format-neon-op op dtype signed (first qs))
                         #:after-last ");\n")))

  (define format (if assembly format-assembly format-intrin))
  (define format-imm (if assembly format-assembly-imm format-intrin-imm))
  (define signed #t)
  (define register->name (make-hash))

  (add-inputs register->name inputs)
  
  (if (list? prgm)
      (for ([isn (in-list prgm)] [i (in-naturals)])
        (match isn
          [(vcnt q dtype rd ra)     (format "vcnt" i register->name (list q q) 8bit 8bit signed rd ra)]
          [(vlshf q dtype rd ra imm)   (format-imm "vshl" i register->name (list q q) dtype dtype signed rd ra imm)]
          [(vand q dtype rd ra rb) (format "vand" i register->name (list q q q) dtype dtype signed rd ra rb)]
          [(vadd q dtype rd ra rb) (format "vadd" i register->name (list q q q) dtype dtype signed rd ra rb)]
          [(vsub q dtype rd ra rb) (format "vsub" i register->name (list q q q) dtype dtype signed rd ra rb)]
          [(vbic q dtype rd ra rb) (format "vbic" i register->name (list q q q) dtype dtype signed rd ra rb)]
          [(vaddl dtype rd ra rb) (format "vaddl" i register->name (list Q D D) dtype (promote dtype) signed rd ra rb)]
          [(vpadd dtype rd ra rb)  (format "vpadd" i register->name (list D D D) dtype dtype signed rd ra rb)]
          [(vpaddl q dtype rd ra)  (format "vpaddl" i register->name (list q q) dtype (promote dtype) signed rd ra)]
          [(vpadal q dtype rd ra)  (format "vpadal" i register->name (list q q q) dtype (promote dtype) signed rd rd ra)]
          [_ (printf " %v " isn)]))
      (println "")))


(define (add-inputs register->name inputs)
  (for ([input inputs])
    (define reg-size (name-map-type input))
    (define id (name-map-name input))
    (define r (name-map-register input))
    (if (equal? reg-size D)
        (hash-set! register->name r (var D id))
        (begin
          ; For Q registers need to map both the D registers to the name
          (define d-reg (bvdbl r))
          (hash-set! register->name d-reg (var Q id))
          (hash-set! register->name (bvadd d-reg (reg 1)) (var Q id))))))
    
;;;;;; NEON intrinsic formatting
; ex.vpadd_u8
(define (dtype->string dtype)
    (cond
      [(bveq dtype 8bit)   "8"]
      [(bveq dtype 16bit) "16"]))

(define (format-neon-op op dtype signed q)
  (define sign-str (if signed "_s" "_u"))
  (define dtype-str (dtype->string dtype))
  (define quad-str (if q "q" ""))
  (define suffix (if (equal? op "vshl") "_n" "")) ; For vshlq -> vshlq_n
  (string-join (list op quad-str suffix sign-str dtype-str "(") ""))

; ex. uint8x8_t
(define (format-neon-output output-id dtype signed q)
  (define dtype-val
    (cond
      [(bveq dtype 8bit)   8]
      [(bveq dtype 16bit) 16]))
  (define register-length (if q 128 64))
  (define lane-val (/ register-length dtype-val))
  (define signed-str (if signed "int" "uint"))
  (define dtype-str (dtype->string dtype))
  (define lane-str (number->string lane-val))
  (string-join (list signed-str dtype-str "x" lane-str "_t " output-id " = ") ""))

; maps register to a variable like x5, x10
(define (format-intrin-input register->name reg q signed dtype)
  (define d-reg (if q (bvdbl reg) reg))
  (define sign (if signed "_s" "_u"))
  (define dtype-str (dtype->string dtype))
  (cond
    [(hash-has-key? register->name d-reg)
     (let ([v (hash-ref register->name d-reg)])
       (define id (var-id v))
       (cond
         [(equal? q (var-type v))
           id]
         [(and (equal? q Q) (equal? (var-type v) D))
          (let ([next-var (hash-ref register->name (bvadd d-reg (bv 1 6)))])
            (string-join (list "vcombine" sign dtype-str "(" id ", " (var-id next-var)) "" #:after-last ")"))]
         [else
          (define part (if (odd? d-reg) "vget_high" "vget_low"))
          (string-join (list part sign dtype-str "(" id ")") "")]))]
    [else
     (let ([new-id (string-append "x" (number->string (bitvector->natural d-reg)))])
       (hash-set! register->name reg (var q new-id))
       new-id)]))


;;;;;;; Assembly formating
(define (format-reg q ra)
  (define number (number->string (bitvector->natural ra)))
  (if q (string-join (list "Q" number) "")
      (string-join (list "D" number) "")))

(define (format-op instr dtype)
  (define dtype-str
    (cond 
      [(bveq dtype 8bit)   ".i8"]
      [(bveq dtype 16bit) ".i16"]))
  (string-join (list instr dtype-str " ") ""))

(define (format-assembly op i x qs dtype output-dtype signed . regs )
  (define operand-list (for/list ([q qs] [reg regs])
    (format-reg q reg)))
  (printf (string-join operand-list
                       ", "
                       #:before-first (format-op op dtype)
                       #:after-last "\n")))

(define (format-assembly-imm op i x qs dtype output-dtype signed dst-reg src-reg imm)
  (define imm-str (number->string (bitvector->natural imm)))
  (define operand-list (list (format-reg (first qs) dst-reg) (format-reg (second qs) src-reg) imm-str))
  (printf (string-join operand-list
                       ", "
                       #:before-first (format-op op dtype)
                       #:after-last "\n")))
