/**
  Query 1: Single employee's schedule for 6 week period
 */
USE cmpsc431;
SELECT
  DATE_ADD(week.start_date, INTERVAL shift.day_ID - 1 DAY) AS 'shiftDate',
  weekday.day_name                                         AS 'day',
  shift_time.shift_start                                   AS 'start',
  shift_time.shift_end                                     AS 'end',
  department.dept_name                                     AS 'department',
  role.role                                                AS 'role'
FROM shift, week, weekday, shift_time, department, role, certification, employee
WHERE shift.day_ID = weekday.day_ID
      AND shift.time_ID = shift_time.time_ID
      AND shift.dept_ID = department.dept_ID
      AND shift.week_ID = week.week_ID
      AND shift.emp_ID = employee.emp_ID
      AND employee.emp_ID = certification.emp_ID
      AND certification.role_ID = role.role_ID
      AND week.week_ID >= 1
      AND week.week_ID <= 6;

/**
  Query 2: Department's Needs for 1 week
 */
USE cmpsc431;
SELECT
  DATE_ADD(week.start_date, INTERVAL department_need.day_ID - 1 DAY) AS 'shiftDate',
  weekday.day_name                                                   AS 'day',
  shift_time.shift_start                                             AS 'start',
  shift_time.shift_end                                               AS 'end',
  department.dept_name                                               AS 'department',
  role.role                                                          AS 'role',
  department_need.need                                               AS 'needs'
FROM department_need, week, weekday, shift_time, department, role
WHERE department_need.week_ID = 1
      AND department_need.week_ID = week.week_ID
      AND department_need.day_ID = weekday.day_ID
      AND department_need.time_ID = shift_time.time_ID
      AND department_need.dept_ID = department.dept_ID
      AND department_need.role_ID = role.role_ID;

/**
  Query 3: Department Weekly Schedules
 */
USE cmpsc431;
SELECT
  department.dept_name                                     AS 'department',
  employee.fname                                           AS 'firstName',
  employee.lname                                           AS 'lastName',
  employee.phone1                                          AS 'phoneNumber',
  DATE_ADD(week.start_date, INTERVAL shift.day_ID - 1 DAY) AS 'shiftDate',
  weekday.day_name                                         AS 'day',
  shift_time.shift_start                                   AS 'start',
  shift_time.shift_end                                     AS 'end'
FROM department, week, employee, shift, shift_time, weekday
WHERE week.week_ID = 1
      AND shift.emp_ID = employee.emp_ID
      AND shift.dept_ID = department.dept_ID
      AND shift.time_ID = shift_time.time_ID
      AND shift.week_ID = week.week_ID
      AND shift.day_ID = weekday.day_ID
ORDER BY department, lastName, firstName;

/**
  Query 4: Employee Weekly Cost
 */
USE cmpsc431;
SELECT
  department.dept_name                                             AS 'department',
  employee.fname                                                   AS 'firstName',
  employee.lname                                                   AS 'lastName',
  DATE_ADD(week.start_date, INTERVAL shift.day_ID - 1 DAY)         AS 'shiftDate',
  shift_time.shift_start                                           AS 'start',
  shift_time.shift_end                                             AS 'end',
  employee.pay_rate * shift.pay_modifier * shift_time.shift_length AS 'pay',
  shift_status.status                                              AS 'status'
FROM department, employee, shift, shift_time, week, shift_status
WHERE DATE_ADD(week.start_date, INTERVAL shift.day_ID - 1 DAY) >= '2017-10-01'
      AND DATE_ADD(week.start_date, INTERVAL shift.day_ID - 1 DAY) <= '2017-10-10'
      AND shift.emp_ID = employee.emp_ID
      AND shift.dept_ID = department.dept_ID
      AND shift.time_ID = shift_time.time_ID
      AND shift.week_ID = week.week_ID
      AND shift.status_ID = shift_status.status_ID
ORDER BY shiftDate, department, lastName, firstName;

/**
  Query 5: List of tables and their descriptions
 */
USE cmpsc431;
SHOW tables;
DESCRIBE address;
DESCRIBE certification;
DESCRIBE department;
DESCRIBE department_need;
DESCRIBE employee;
DESCRIBE role;
DESCRIBE shift;
DESCRIBE shift_status;
DESCRIBE shift_time;
DESCRIBE week;
DESCRIBE weekday;
 
 
/**
  Query 6: All data
 */
USE cmpsc431;
SELECT * FROM address;
SELECT * FROM certification;
SELECT * FROM department;
SELECT * FROM department_need;
SELECT * FROM employee;
SELECT * FROM role;
SELECT * FROM shift;
SELECT * FROM shift_status;
SELECT * FROM shift_time;
SELECT * FROM week;
SELECT * FROM weekday;