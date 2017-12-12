import csv

## Load data
with open('../outputs/query1.out', 'r') as file:
    input = csv.reader(file)
    query1 = list(input)

with open('../outputs/query2.out', 'r') as file:
    input = csv.reader(file)
    query2 = list(input)

with open('../outputs/query3.out', 'r') as file:
    input = csv.reader(file)
    query3 = list(input)

with open('../outputs/query4.out', 'r') as file:
    input = csv.reader(file)
    query4 = list(input)

## Query 1 Formatting
# Set up values
q1_header = '|{0:^22}|{1:^22}|{2:^22}|{3:^22}|{4:^22}|{5:^22}|{6:^22}|'.format('Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday')
q1_title = (query1[1][4] + ' ' + query1[1][5] + '\'s Schedule for the week of ' + query1[1][-1] + ' to ' + query1[-1][-1])
q1_width = len(q1_header)

# Print
print('-' * q1_width)
print('|{0:^{1}s}|'.format(q1_title,q1_width-2))
print('-' * q1_width)
print(q1_header)
print('-' * q1_width)

for week in range(0,len(query1),5):
    sun1,mon1,tue1,wed1,thu1,fri1,sat1 = '','','','','','',''
    sun2,mon2,tue2,wed2,thu2,fri2,sat2 = 'OFF','OFF','OFF','OFF','OFF','OFF','OFF'
    sun3,mon3,tue3,wed3,thu3,fri3,sat3 = '','','','','','',''
    for day in range(0,5):
        shift = query1[week+day]
        line1 = shift[0]
        line2 = (shift[2] + ' - ' + shift[3])
        line3 = (shift[6] + ',' + shift[7])
        if shift[1] == 'SUN':
            sun1 = line1
            sun2 = line2
            sun3 = line3
        if shift[1] == 'MON':
            mon1 = line1
            mon2 = line2
            mon3 = line3
        if shift[1] == 'TUE':
            tue1 = line1
            tue2 = line2
            tue3 = line3
        if shift[1] == 'WED':
            wed1 = line1
            wed2 = line2
            wed3 = line3
        if shift[1] == 'THU':
            thu1 = line1
            thu2 = line2
            thu3 = line3
        if shift[1] == 'FRI':
            fri1 = line1
            fri2 = line2
            fri3 = line3
        if shift[1] == 'SAT':
            sat1 = line1
            sat2 = line2
            sat3 = line3
    print('|{0:^22}|{1:^22}|{2:^22}|{3:^22}|{4:^22}|{5:^22}|{6:^22}|'.format(sun1,mon1,tue1,wed1,thu1,fri1,sat1))
    print('|{0:^22}|{1:^22}|{2:^22}|{3:^22}|{4:^22}|{5:^22}|{6:^22}|'.format(sun2,mon2,tue2,wed2,thu2,fri2,sat2))
    print('|{0:^22}|{1:^22}|{2:^22}|{3:^22}|{4:^22}|{5:^22}|{6:^22}|'.format(sun3,mon3,tue3,wed3,thu3,fri3,sat3))
    print('-' * q1_width)

## Query 2 Formatting
# Set up values
q2_header = '|{0:^12}|{1:^5}|{2:^13}|{3:^11}|{4:^12}|{5:^7}|{6:^6}|'.format('Date', 'Day', 'Shift Start', 'Shift End', 'Department', 'Role', 'Need')
q2_title =  query2[0][4] + " " + "needs for the week of " + query2[0][0]
q2_width = len(q2_header)

# Print
print('-' * q2_width)
print('|{0:^{1}s}|'.format(q2_title,q2_width-2))
print('-' * q2_width)
print(q2_header)
print('-' * q2_width)
for need in query2:
    print('|{0:^12}|{1:^5}|{2:^13}|{3:^11}|{4:^12}|{5:^7}|{6:^6}|'.format(need[0],need[1],need[2],need[3],need[4],need[5],need[6]))
print('-' * q2_width)

## Query 3 Formatting
# Set up values
q3_header = '|{0:^12}|{1:^5}|{2:^15}|{3:^15}|{4:^14}|{5:^10}|{6:^10}|'.format('Date', 'Day', 'Last Name', 'First Name', 'Phone Number', 'Shift Start', 'Shift End')
q3_title =  query3[0][0] + " " + "schedule for the week of " + query3[0][4]
q3_width = len(q3_header)

# Print
print('-' * q3_width)
print('|{0:^{1}s}|'.format(q3_title,q3_width-2))
print('-' * q3_width)
print(q3_header)
print('-' * q3_width)
for shift in query3:
    print('|{0:^12}|{1:^5}|{2:^15}|{3:^15}|{4:^14}|{5:^10}|{6:^10}|'.format(shift[4],shift[5],shift[2],shift[1],shift[3],shift[6],shift[7]))
print('-' * q3_width)

## Query 4 Formatting
# Set up values
q4 = sorted(query4, key = lambda x: (x[0],x[3],x[4]))
q4_header = '|{0:^15}|{1:^15}|{2:^15}|{3:^15}|{4:^7}|'.format('Department','Date','Shift Start','Shift End', 'Cost')
q4_title = ("Department costs per shift between " + query4[0][3] + " and " + query4[-1][3])
q4_width = len(q4_header)
prevDept = q4[0][0]
prevDate = q4[0][3]
prevStart = q4[0][4]
prevEnd = q4[0][5]
cost = 0.0;

# Print
print('-' * q4_width)
print('|{0:^{1}s}|'.format(q4_title,q4_width-2))
print('-' * q4_width)
print(q4_header)
print('-' * q4_width)
while q4:
    shift = q4.pop(0)
    if ( shift[0] == prevDept and shift[3] == prevDate and shift[4] == prevStart  and shift[5] == prevEnd):
         cost = cost + float(shift[6]) * float(shift[7])
    else:
        print('|{0:^15}|{1:^15}|{2:^15}|{3:^15}|{4:^7}|'.format(prevDept,prevDate,prevStart,prevEnd,str(cost)))
        cost = float(shift[6]) * float(shift[7])
        prevDept = shift[0]
        prevDate = shift[3]
        prevStart = shift[4]
        prevEnd = shift[5]
print('-' * q4_width)
