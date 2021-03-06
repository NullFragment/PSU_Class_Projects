SELECT fname, lname, dname, shift_start, shift_end, day_of_week
FROM week, shift, department, shift_times
WHERE week.startDate = start_of_week_date
AND shift.STID = shift_times.STID
AND shift.WID = week.WID
AND shift.EID = employee.EID
AND shift.DID = department.DID
ORDER BY day_of_week, lname, fname