clear;clc

fprintf('---------------PROBLEM 1---------------\n\n')
I = 3.50017;
fprintf('----------Trapezoidal Rule----------\n')
TR_n1 = TrapezoidalRule(@func1, 0, 4, 1)
TR_n1_error = abs((I-TR_n1)/I)
TR_n2 = TrapezoidalRule(@func1, 0, 4, 2)
TR_n2_error = abs((I-TR_n2)/I)
TR_n4 = TrapezoidalRule(@func1, 0, 4, 4)
TR_n4_error = abs((I-TR_n4)/I)

fprintf('\n----------Simpsons 1/3 Rule----------\n')
S13_n1 = Simpsons13(@func1, 0, 4, 2)
S13_n1_error = abs((I-S13_n1)/I)
S13_n4 = Simpsons13(@func1, 0, 4, 4)
S13_n4_error = abs((I-S13_n4)/I)

fprintf('\n----------Simpsons 3/8 Rule----------\n')
S38_n1 = Simpsons38(@func1, 0, 4, 3)
S33_n1_error = abs((I-S38_n1)/I)

fprintf('\n----------Two Point Rule----------\n')
TP = TwoPointRule(@func1, 0, 4)
TP_error = abs((I-TP)/I)

fprintf('\n\n---------------PROBLEM 2---------------\n\n')
i = 1;
for t = 0:0.1:2
    Vals(i,:) = ode(t);
    time(i) = t;
    i = i + 1;
end

fprintf('\n---------- Euler Method----------\n')
EulerValues_2s = EulerMethod(@ode, [pi/4, 0], 0, 2.1, 0.1)
EulerValues_10s = EulerMethod(@ode, [pi/4, 0], 0, 10.1, 0.1)

fprintf('\n---------- Heun Method----------\n')
HeunValues_2s = HeunMethod(@ode, [pi/4, 0], 0, 2.1, 0.1)
HeunValues_10s = HeunMethod(@ode, [pi/4, 0], 0, 10.1, 0.1)

fprintf('\n---------- RK4 Method----------\n')
RK4Values_2s = RK4Method(@ode, [pi/4, 0], 0, 2.1, 0.1)
RK4Values_10s = RK4Method(@ode, [pi/4, 0], 0, 10.1, 0.1)

%---------------plot------------------------
figure(1)
plot(time',Vals(:,1))
hold all
plot(time',EulerValues_2s(:,1))
hold all
plot(time',HeunValues_2s(:,1))
hold all
plot(time',RK4Values_2s(:,1))
hold all
legend('Analytic','Euler', 'Heun', 'RK4')

hold all
figure(2)
plot(time',Vals(:,2))
hold all
plot(time',EulerValues_2s(:,2))
hold all
plot(time',HeunValues_2s(:,2))
hold all
plot(time',RK4Values_2s(:,2))
hold all
legend('Analytic','Euler', 'Heun', 'RK4')

hold all
figure(3)
plot(EulerValues_10s(:,1),EulerValues_10s(:,2)) 
hold all
plot(HeunValues_10s(:,1),HeunValues_10s(:,2))
hold all
plot(RK4Values_10s(:,1),RK4Values_10s(:,2)) 
hold all
legend('Euler', 'Heun', 'RK4')
