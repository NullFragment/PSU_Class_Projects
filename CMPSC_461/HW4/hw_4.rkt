; PROBLEM 1
(define (fib n)
  (cond
    ((= n 0) 0)
    ((= n 1) 1)
    ((> n 1) (+ (fib (- n 1)) (fib (- n 2))))
    )
  )

; PROBLEM 2
(define (M n)
  (cond
    ((= n 0) 0)
    ((> n 0) (- n (F (M (- n 1)))))
    )
  )

(define (F n)
  (cond
    ((= n 0) 1)
    ((> n 0) (- n (M (F (- n 1)))))
    )
  )

; PROBLEM 3
(define (my_gcd x y)
  (cond
    ((= y 0) x)
    ((> y 0) (my_gcd y (remainder x y)))
    )
  )

; PROBLEM 4
(define (addone x) (+ x 1))

(define (ncall n f x)
  (cond
    ((= n 0) x)
    ((> n 0) (f (ncall (- n 1) f x)))
    )
  )



'Problem\ 1:
(fib 20)

(newline)
'Problem\ 2:
(F 0)
(M 0)
(F 1)
(M 1)

(newline)
'Problem\ 3:
'Mine:
(my_gcd 100 12)
'Built-in:
(gcd 100 12)

(newline)
'Problem\ 4:
(ncall 0 addone 1)
(ncall 1 addone 1)
(ncall 2 addone 1)
(ncall 4 addone 2)