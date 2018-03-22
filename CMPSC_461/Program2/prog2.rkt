; PROBLEM 1
(define (remove-if fun list_in)
  (cond
    ((not (list? list_in)) 'ERR:NOT_A_LIST)
    ((null? list_in) list_in) 
    ((equal? (fun (car list_in)) #t) (remove-if fun (cdr list_in)))
    (else (cons (car list_in) (remove-if fun (cdr list_in))))))


; PROBLEM 2
(define (removeLast list_in)
  (cond
    ((null? list_in) '())
    ((null? (cdr list_in)) (cdr list_in))
    (else (cons (car list_in) (removeLast (cdr list_in))))))


; PROBLEM 3
(define (reverseList list_in) (rev list_in '()))

(define (rev list_1 list_2)
  (cond
    ((null? list_1) list_2)
    (else (append (rev (cdr list_1) (list (car list_1))) list_2))))


(define (small_nums x)
  (cond
    ((= 0 x) '(zero))
    ((= 1 x) '(one))
    ((= 2 x) '(two))
    ((= 3 x) '(three))
    ((= 4 x) '(four))
    ((= 5 x) '(five))
    ((= 6 x) '(six))
    ((= 7 x) '(seven))
    ((= 8 x) '(eight))
    ((= 9 x) '(nine))
    ((= 10 x) '(ten))
    ((= 11 x) '(eleven))
    ((= 12 x) '(twelve))
    ((= 13 x) '(thirteen))
    ((= 14 x) '(fourteen))
    ((= 15 x) '(fifteen))
    ((= 16 x) '(sixteen))
    ((= 17 x) '(seventeen))
    ((= 18 x) '(eighteen))
    ((= 19 x) '(nineteen))))

(define (large_nums x)
  (cond
    ((= 2 x) '(twenty))
    ((= 3 x) '(thirty))
    ((= 4 x) '(forty))
    ((= 5 x) '(fifty))
    ((= 6 x) '(sixty))
    ((= 7 x) '(seventy))
    ((= 8 x) '(eighty))
    ((= 9 x) '(ninety))))

(define (int-to-words x)
  (cond
    ((> 20 x) (small_nums x))
    ((= (remainder x 10) 0) (large_nums (quotient x 10)))
    (else (append (large_nums (quotient x 10)) (small_nums (remainder x 10))))))


; PROBLEM 5
; PART A
(define (nzero n)
  (cond
    ((= n 0) '())
    ((= n 1) '(0))
    (else (cons 0 (nzero (- n 1))))))

; PART B
(define (polyAdd poly1 poly2)
  (cond
    ((and (= 0 (length poly1)) (= 0 (length poly2))) '())
    ((and (= 0 (length poly1)) (< 0 (length poly2))) poly2)
    ((and (< 0 (length poly1)) (= 0 (length poly2))) poly1)
    ((and (= 1 (length poly1)) (= 1 (length poly2)))
      (list (+ (car poly1) (car poly2))))
    (else
      (cons (+ (car poly1) (car poly2)) (polyAdd (cdr poly1) (cdr poly2))))))

; PART C
(define (polyAddList polyList)
  (cond
    ((= 0 (length polyList)) '())
    ((= 1 (length polyList)) (car polyList))
    ((< 1 (length polyList)) (polyAdd (car polyList) (polyAddList (cdr polyList))))))

; PART D
(define (listMult n list)
  (map (lambda (L) (* n L)) list))
  
(define (polyMultHelper poly1 poly2 n)
  (cond
    ((or (= 0 (length poly1)) (= 0 (length poly2))) '())
    (else (cons
           (listMult (car poly1) (append (nzero n) poly2))
           (polyMultHelper (cdr poly1) poly2 (+ n 1))))))

(define (polyMult poly1 poly2)
  (polyAddList (polyMultHelper poly1 poly2 0)))





;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;         Test Cases : Alexandar Devic          ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Test Part 1
(display "Testing Part 1 : remove-if")(newline)
(if (equal? (remove-if (lambda (x) (> x 3)) '(10 1 7 2)) '(1 2)) 'pass '--fail--)
(if (equal? (remove-if (lambda (x) (equal? x '(1 2 3))) '(10 1 (1 2) 2 10 (1 2 3))) '(10 1 (1 2) 2 10)) 'pass '--fail--)
(if (equal? (remove-if (lambda (x) (equal? x '())) '(() () () 2)) '(2)) 'pass '--fail--)
(if (equal? (remove-if (lambda (x) #t) '(10 1 7 2)) '()) 'pass '--fail--)
(if (equal? (remove-if (lambda (x) #f) '(10 1 7 2)) '(10 1 7 2)) 'pass '--fail--)

;;; Test Part 2
(display "Testing Part 2 : removeLast")(newline)
(if (equal? (removeLast '(1 2 3 4)) '(1 2 3)) 'pass '--fail--)
(if (equal? (removeLast '(4)) '()) 'pass '--fail--)
(if (equal? (removeLast '((1 2 3) (1 2))) '((1 2 3))) 'pass '--fail--)
(if (equal? (removeLast '()) '()) 'pass '--fail--)
(if (equal? (removeLast '(() 12 (2 3 4) (1 2))) '(() 12 (2 3 4))) 'pass '--fail--)

;;; Test Part 3a
(display "Testing Part 3a : rev")(newline)
(if (equal? (rev '(1 2 3) '(4 5)) '(3 2 1 4 5)) 'pass '--fail--)
(if (equal? (rev '(1 2) '(4 5 6)) '(2 1 4 5 6)) 'pass '--fail--)
(if (equal? (rev '() '(4 5)) '(4 5)) 'pass '--fail--)
(if (equal? (rev '(1 2 3) '()) '(3 2 1)) 'pass '--fail--)
(if (equal? (rev '(1 (1 2 3) (4 5 6) (2) 0) '((4) 5)) '(0 (2) (4 5 6) (1 2 3) 1 (4) 5)) 'pass '--fail--)

;;; Test Part 3b
(display "Testing Part 3b : reverseList")(newline)
(if (equal? (reverseList '(1 2 3)) '(3 2 1)) 'pass '--fail--)
(if (equal? (reverseList '(4 (2) 2)) '(2 (2) 4)) 'pass '--fail--)
(if (equal? (reverseList '()) '()) 'pass '--fail--)
(if (equal? (reverseList '(1)) '(1)) 'pass '--fail--)
(if (equal? (reverseList '(1 (2 3 4) (5 6) ())) '(() (5 6) (2 3 4) 1)) 'pass '--fail--)

;;; Test Part 4
(display "Testing Part 4 : int-to-words")(newline)
(if (equal? (int-to-words 13) '(thirteen)) 'pass '--fail--)
(if (equal? (int-to-words 42) '(forty two)) 'pass '--fail--)
(if (equal? (int-to-words 0) '(zero)) 'pass '--fail--)
(if (equal? (int-to-words 10) '(ten)) 'pass '--fail--)
(if (equal? (int-to-words 30) '(thirty)) 'pass '--fail--)

;;; Test Part 5a
(display "Testing Part 5a : nzero")(newline)
(if (equal? (nzero 3) '(0 0 0)) 'pass '--fail--)
(if (equal? (nzero 10) '(0 0 0 0 0 0 0 0 0 0)) 'pass '--fail--)
(if (equal? (nzero 1) '(0)) 'pass '--fail--)
(if (equal? (nzero 5) '(0 0 0 0 0)) 'pass '--fail--)
(if (equal? (nzero 0) '()) 'pass '--fail--)

;;; Test Part 5b
(display "Testing Part 5b : polyAdd")(newline)
(if (equal? (polyAdd '(1 2 1) '(0 2 4 2)) '(1 4 5 2)) 'pass '--fail--)
(if (equal? (polyAdd '(0 2 4 2) '(1 2 1)) '(1 4 5 2)) 'pass '--fail--)
(if (equal? (polyAdd '() '(1 1 1 1)) '(1 1 1 1)) 'pass '--fail--)
(if (equal? (polyAdd '(1 1 1 1) '()) '(1 1 1 1)) 'pass '--fail--)
(if (equal? (polyAdd '(9 100 2 3 4 5 6 7 8 9) '(3 101 0 3)) '(12 201 2 6 4 5 6 7 8 9)) 'pass '--fail--)

;;; Test Part 5c
(display "Testing Part 5c : polyAddList")(newline)
(if (equal? (polyAddList '((1 2 1) (0 2 4 2))) '(1 4 5 2)) 'pass '--fail--)
(if (equal? (polyAddList '((1 2 1) (0 2 4 2) (0 0 1 2 1))) '(1 4 6 4 1)) 'pass '--fail--)
(if (equal? (polyAddList '()) '()) 'pass '--fail--)
(if (equal? (polyAddList '((1 2 3))) '(1 2 3)) 'pass '--fail--)
(if (equal? (polyAddList '((0))) '(0)) 'pass '--fail--)

;;; Test Part 5d
(display "Testing Part 5d : polyMult")(newline)
(if (equal? (polyMult '(1 2 1) '(1 2 1)) '(1 4 6 4 1)) 'pass '--fail--)
(if (equal? (polyMult '(0 1) '(1 2 1 7)) '(0 1 2 1 7)) 'pass '--fail--)
(if (equal? (polyMult '(3) '(1 2 1 7)) '(3 6 3 21)) 'pass '--fail--)
(if (equal? (polyMult '(0) '(3 56 2 7 3 7 1 7 2)) '(0 0 0 0 0 0 0 0 0)) 'pass '--fail--)
(if (equal? (polyMult '(1 0 1 4 9 1) '(3 1 0 0 0 1)) '(3 1 3 13 31 13 1 1 4 9 1)) 'pass '--fail--)
