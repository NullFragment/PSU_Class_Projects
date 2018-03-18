(define testlist '(1 2 3 4 5))

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
    ((null? (cdr list_in)) (cdr list_in))
    (else (cons (car list_in) (removeLast (cdr list_in))))))


; PROBLEM 3
(define (reverseList list_in)
  (cond
    ((null? list_in) '())
    (else (append (reverseList (cdr list_in)) (list (car list_in))))))

(define (rev list_1 list_2) (append (reverseList list_1) list_2))

; Problem 3
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



'------------Test\ 1------------
'-------Expecting:\(1\ 2\)--------
(remove-if (lambda (x) (> x 3)) '(10 1 7 2))

'------------Test\ 2------------
'------Expecting:\(1\ 2\ 3\)-------
(removeLast '(1 2 3 4))

'------------Test\ 3------------
'----Expecting:\(3\ 2\ 1\ 4\ 5\)-----
(rev '(1 2 3) '(4 5))

'------------Test\ 4------------
(int-to-words 10)
(int-to-words 20)
(int-to-words 30)
(int-to-words 45)

'------------Test\ 5------------
'----Expecting:\(1\ 4\ 6\ 4\ 1\)-----
(polyMult '(1 2 1) '(1 2 1))
