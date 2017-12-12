from random import randint, sample
from datetime import datetime, timedelta
start = datetime.strptime('20171001', '%Y%m%d').date()
addition = -1;

for emp_ID in range(1, 21):
    for week_ID in range(1, 13):
        addition = (week_ID - 1) * 7;
        for day_ID in sample(range(1, 8), 5):
            addition2 = addition + (day_ID - 1);
            pay_modifier = 1.0
            if day_ID == 1 or day_ID == 7:
                time_ID = randint(4, 5)
                pay_modifier += 0.5
            else:
                time_ID = randint(1, 3)
            dept_ID = randint(1, 6)
            status_ID = randint(1, 4)
            if status_ID == 1:
                pay_modifier += 0.5
            if status_ID == 2:
                pay_modifier = 0.0
            print("INSERT INTO shift ,"
                  + str(emp_ID) + ","
                  + str(dept_ID) + ","
                  + str(time_ID) + ","
                  + str(week_ID) + ","
                  + str(day_ID) + ","
                  + str(status_ID) + ","
                  + str(pay_modifier) + ","
                  + str(start + timedelta(days=addition2)))
