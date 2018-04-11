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
    ((equal? (caar assoc_list) k) (car assoc_list))
    (else (lookup k (cdr assoc_list)))))


#| PROBLEM 1 |#
(define al '())

(define (fac x)
  (cond
    ((= x 1) 1)
    (else (* x (fac (- x 1))))))

(define (fac_mem x)
  (let ((res (lookup x al)))
    (cond
      ((equal? res #f) (begin (set! al (bind x (fac x) al))) (fac x))
      (else (begin (display "memoization hit: ")) (cadr res))
      )))


#| PROBLEM 2 |# 
(define (build_mem f)
  (let ((local_al '()))
    (lambda (n)
      (let ((res (lookup n local_al)))
        (cond
          ((equal? res #f) (begin (set! local_al (bind n (f n) local_al))) (f n))
          (else (begin (display "memoization hit: ")) (cadr res))))
      )
    ))
