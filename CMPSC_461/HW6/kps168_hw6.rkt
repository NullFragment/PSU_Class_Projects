#|
    WRITTEN BY: Kyle Salitrik
    PSU ID: kps168
    ASSIGNMENT: HW 6
|#


#| GLOBALLY USED FUNCTIONS |#
(define (bind k v assoc_list) (cons (list k v) assoc_list))
(define (lookup k assoc_list)
  (cond
    ((null? assoc_list) #f)
    ((equal? (caar assoc_list) k) (cadar assoc_list))
    (else (lookup k (cdr assoc_list)))))


#| PROBLEM 1 |#
(define al '())

(define (fac x)
  (cond
    ((= x 0) 1)
    (else (* x (fac (- x 1))))))

(define (fac_mem x)
  (let ((res (lookup x al)))
    (cond
      ((equal? res #f) (begin (set! al (bind x (fac x) al))) (fac x))
      (else (begin (display "memoization hit: ")) res)
      )))


#| PROBLEM 2 |# 
(define (build_mem f)
  (let ((local_al '()))
    (lambda (n)
      (let ((res (lookup n local_al)))
        (cond
          ((equal? res #f) (begin (set! local_al (bind n (f n) local_al))) (f n))
          (else (begin (display "memoization hit: ")) res)))
      )
    ))

#| TEST CASES TAKEN FROM PIAZZA |#
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;         Test Cases : Alexandar Devic          ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Test Part 1a
(display "Testing Part 1a : fac")(newline)
(if (= (fac 5) 120) 'pass 'fail)
(if (= (fac 10) 3628800) 'pass 'fail)
(if (= (fac 0) 1) 'pass 'fail)

;;; Test Part 1b
(display "Testing Part 1a : bind / lookup")(newline)
(define al '())
(if (equal? (lookup 1 (bind 1 123 al)) 123) 'pass 'fail)
(if (equal? (lookup 1 (bind 2 123 al)) #f) 'pass 'fail)
(if (null? al) 'pass 'fail)

;;; Test Part 1c
(display "Testing Part 1c : fac_mem")(newline)
(if (= (fac_mem 1) 1) 'pass 'fail)
(if (= (fac_mem 10) 3628800) 'pass 'fail)
(if (= (fac_mem 5) 120) 'pass 'fail)
(display "The following should be memoization hits!")(newline)
(if (= (fac_mem 1) 1) 'pass 'fail)
(if (= (fac_mem 10) 3628800) 'pass 'fail)
(if (= (fac_mem 5) 120) 'pass 'fail)

;;; Test Part 2
(display "Testing Part 2 : build_mem")(newline)
; Some helper functions to be memoized
(define (female n) (cond ((= n 0) 1)((> n 0) (- n (male (female (- n 1)))))(else 'error)))
(define (male n) (cond ((= n 0) 0)((> n 0) (- n (female (male (- n 1)))))(else 'error)))
(define (fib n) (cond ((= n 0) 0)((= n 1) 1)((> n 1) (+ (fib (- n 1)) (fib (- n 2))))(else 'error)))

(define memFac (build_mem fac))
(define memF (build_mem female))
(define memM (build_mem male))
(define memFib (build_mem fib))


(if (= (memFac 5) 120) 'pass 'fail)
(if (= (memFac 10) 3628800) 'pass 'fail)
(if (= (memFac 0) 1) 'pass 'fail)
(if (= (memF 73) 45) 'pass 'fail)
(if (= (memM 51) 32) 'pass 'fail)
(if (= (memFib 20) 6765) 'pass 'fail)


(display "The following should be memoization hits!")(newline)
(if (= (memFac 5) 120) 'pass 'fail)
(if (= (memFac 10) 3628800) 'pass 'fail)
(if (= (memFac 0) 1) 'pass 'fail)
(if (= (memF 73) 45) 'pass 'fail)
(if (= (memM 51) 32) 'pass 'fail)
(if (= (memFib 20) 6765) 'pass 'fail)