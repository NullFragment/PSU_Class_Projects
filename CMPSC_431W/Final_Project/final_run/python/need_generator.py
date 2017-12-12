from random import randint, sample
from datetime import datetime, timedelta
start = datetime.strptime('20171001', '%Y%m%d').date()
addition = -1;

for week_ID in range(1, 13):
    for day_ID in range(1, 8):
        addition = addition+1;
        if day_ID == 1 or day_ID == 7:
            for time_ID in range(4, 6):
                for dept_ID in range(1, 7):
                    for role_ID in range(1, 6):
                        print("INSERT INTO department_need ,"
                              + str(week_ID) + ","
                              + str(day_ID) + ","
                              + str(time_ID) + ","
                              + str(dept_ID) + ","
                              + str(role_ID) + ","
                              + str(randint(2, 3)) + ","
                              + str(start + timedelta(days=addition)))
        else:
            for time_ID in range(1, 4):
                for dept_ID in range(1, 7):
                    for role_ID in range(1, 6):
                        print("INSERT INTO department_need ,"
                              + str(week_ID) + ", "
                              + str(day_ID) + ", "
                              + str(time_ID) + ", "
                              + str(dept_ID) + ", "
                              + str(role_ID) + ", "
                              + str(randint(2, 3)) + ","
                              + str(start + timedelta(days=addition)))
