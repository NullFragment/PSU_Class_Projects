from random import randint, sample

print("INSERT INTO cmpsc431.department_need (week_ID, day_ID, time_ID, dept_ID, role_ID, need) VALUES ")
for week_ID in range(1, 13):
    for day_ID in range(1, 8):
        if day_ID == 1 or day_ID == 7:
            for time_ID in range(4, 6):
                for dept_ID in range(1, 7):
                    for role_ID in range(1, 6):
                        print("("
                              + str(week_ID) + ", "
                              + str(day_ID) + ", "
                              + str(time_ID) + ", "
                              + str(dept_ID) + ", "
                              + str(role_ID) + ", "
                              + str(randint(2, 3)) + "),")
        else:
            for time_ID in range(1, 4):
                for dept_ID in range(1, 7):
                    for role_ID in range(1, 6):
                        print("("
                              + str(week_ID) + ", "
                              + str(day_ID) + ", "
                              + str(time_ID) + ", "
                              + str(dept_ID) + ", "
                              + str(role_ID) + ", "
                              + str(randint(2, 3)) + "),")

print("INSERT INTO cmpsc431.shift (emp_ID, dept_ID, time_ID, week_ID, day_ID, status_ID, pay_modifier) VALUES ")
for emp_ID in range(1, 21):
    for week_ID in range(1, 13):
        for day_ID in sample(range(1, 8), 5):
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
            print("("

                  + str(emp_ID) + ", "
                  + str(dept_ID) + ", "
                  + str(time_ID) + ", "
                  + str(week_ID) + ", "
                  + str(day_ID) + ", "
                  + str(status_ID) + ", "
                  + str(pay_modifier) + "),")
