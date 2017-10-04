USE cmpsc431;

## Weekly Schedule Query
SET @week_start_date = '2017-10-01';
SELECT dept_name, fname, lname, day_name, shift_start, shift_end
FROM week, shift, department, shift_times, employee, weekday
WHERE week.start_date = @week_start_date
      AND shift.employee = employee.EID
      AND shift.week_id = week.WID
      AND shift.department = department.DID
      AND shift.shift_time = shift_times.STID
      AND shift.dow = weekday.WDID
ORDER BY dept_name, lname, fname;

## Department Charge Nurse Query
SET @week_start_date = '2017-10-01';
SELECT dept_name, day_name, fname, lname, shift_start, shift_end
FROM week, shift, department, shift_times, employee, weekday
WHERE week.start_date = @week_start_date
      AND shift.employee = employee.EID
      AND shift.week_id = week.WID
      AND shift.department = department.DID
      AND shift.shift_time = shift_times.STID
      AND shift.dow = weekday.WDID
      AND department.charge_nurse = employee.EID
ORDER BY dept_name, day_name, shift_start;

## Employee Schedule Query
SET @desired_employee = 1;
SET @week_start_date = '2017-10-01';
SELECT dept_name, fname, lname, day_name, shift_start, shift_end, weekday.WDID
FROM week, shift, department, shift_times, employee, weekday
WHERE week.start_date = @week_start_date
      AND employee.EID = @desired_employee
      AND shift.employee = employee.EID
      AND shift.week_id = week.WID
      AND shift.department = department.DID
      AND shift.shift_time = shift_times.STID
      AND shift.dow = weekday.WDID
ORDER BY WDID, shift_start