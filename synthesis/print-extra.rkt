#lang racket

; Function header and data loading + storage

(provide (all-defined-out))

(define (print-imports #:outport [outport (current-output-port)])
  (define imports #<<here-string-delimiter
#include "stdint.h" 
#include "arm_neon.h"

extern "C" int reset(uint16_t* dst) {
    uint16x8_t x = vld1q_u16(dst);
    x = veorq_u16(x, x);
    vst1q_u16(dst, x);
    return 0;
}

here-string-delimiter
    )
  (fprintf outport imports))


(define (print-header k a-bw b-bw #:outport [outport (current-output-port)])
  (define suffix (if (= 16 k) "" "_half"))
  (define input-type (if (= 16 k) "int8x16_t" "int8x8_t"))
  (define method-name (string-join
                       (list"extern \"C\" int update_unipolar_a"
                            (number->string a-bw)
                            "b" (number->string b-bw) suffix)
                       ""))
  (define args "(int8_t* src_a, int8_t* src_b, int16_t* dst, int a_str1, int a_str0, int b_str0) {\n")
  (fprintf outport method-name)
  (fprintf outport args)

  (define (print-out x) (fprintf outport x))
  ; Data loading for A
  (for ([i (in-range a-bw)])
    (fprintf outport "\t~a a~v[8];\n" input-type i))

  (define iters (if (= k 16) 8 4))
  (fprintf outport "\tfor(int i = 0; i < ~v; i++) {\n" iters)
  (for ([i (in-range a-bw)])
    (if (= k 16)
        (fprintf outport "\t\ta~v[i] = vld1q_s8(src_a + i*a_str0 + ~v*a_str1);\n" i i)
        (begin
          (fprintf outport "\t\tint8x16_t aa~v = vld1q_s8(src_a + i*2*a_str0 + ~v*a_str1);\n" i i)
          (fprintf outport "\t\ta~v[2*i] = vget_low_s8(aa~v);\n" i i)
          (fprintf outport "\t\ta~v[2*i + 1] = vget_high_s8(aa~v);\n" i i))))
  (fprintf outport "\t}\n");

  ; Data loading for B (assumes n = 1)
  (define vld (if (= k 16) "vld1q_s8" "vld1_s8"))
  (for ([i (in-range b-bw)])
    (define i-str (number->string i))
    (fprintf outport "\t~a b~v = ~a(src_b + ~v*b_str0);\n" input-type i vld i))


  ; Load destination
  (define output-load "\tint16x8_t acc = vld1q_s16(dst);\n")
  (fprintf outport output-load)
  
  )

(define (print-ending #:outport [outport (current-output-port)])
  (fprintf outport "\tvst1q_s16(dst, out);\n")
  (fprintf outport "\treturn 0;\n")
  (fprintf outport "}\n"))
