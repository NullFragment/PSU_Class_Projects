DROP SCHEMA cmpsc431;
CREATE SCHEMA cmpsc431;
USE cmpsc431;

CREATE TABLE cmpsc431.week
(
    week_ID int(11) PRIMARY KEY NOT NULL AUTO_INCREMENT,
    start_date date NOT NULL,
    end_date date NOT NULL
);

CREATE TABLE cmpsc431.weekday
(
    day_ID int(11) PRIMARY KEY NOT NULL AUTO_INCREMENT,
    day_name char(10)
);

CREATE TABLE cmpsc431.shift_status
(
    status_ID int(11) PRIMARY KEY NOT NULL AUTO_INCREMENT,
    status char(50)
);

CREATE TABLE cmpsc431.shift_time
(
    time_ID int(11) PRIMARY KEY NOT NULL AUTO_INCREMENT,
    shift_start time NOT NULL,
    shift_end time NOT NULL
);

CREATE TABLE cmpsc431.role
(
    role_ID int(11) PRIMARY KEY NOT NULL AUTO_INCREMENT,
    role char(50) NOT NULL
);

CREATE UNIQUE INDEX roles_role_ID_uindex ON cmpsc431.role (role_ID);

CREATE TABLE cmpsc431.department
(
    dept_ID int(11) PRIMARY KEY NOT NULL AUTO_INCREMENT,
    min_staff int(11) DEFAULT '0' NOT NULL,
    max_staff int(11) DEFAULT '100' NOT NULL,
    beds int(11) DEFAULT '0' NOT NULL,
    dept_name char(50) NOT NULL
);

CREATE TABLE cmpsc431.employee
(
    emp_ID int(11) PRIMARY KEY NOT NULL AUTO_INCREMENT,
    home_dept int(11),
    fname char(50) NOT NULL,
    mname char(50),
    lname char(50),
    ssn char(12),
    phone1 char(13),
    phone2 char(13),
    start_date datetime DEFAULT CURRENT_TIMESTAMP NOT NULL,
    end_date date,
    full_time tinyint(1) DEFAULT '0' NOT NULL,
    salaried tinyint(1) DEFAULT '0' NOT NULL,
    pay_rate double DEFAULT '0' NOT NULL,
    CONSTRAINT employee_department_DID_fk FOREIGN KEY (home_dept) REFERENCES department (dept_ID)
);
CREATE INDEX employee_department_DID_fk ON cmpsc431.employee (home_dept);

CREATE TABLE cmpsc431.address
(
    address_ID int(11) PRIMARY KEY NOT NULL AUTO_INCREMENT,
    emp_ID int(11),
    street1 char(50),
    street2 char(50),
    city char(50),
    state char(50),
    zip char(50),
    CONSTRAINT address_employee_EID_fk FOREIGN KEY (emp_ID) REFERENCES employee (emp_ID)
);
CREATE UNIQUE INDEX address_AID_uindex ON cmpsc431.address (address_ID);
CREATE INDEX address_employee_EID_fk ON cmpsc431.address (emp_ID);

CREATE TABLE cmpsc431.certification
(
    cert_ID int(11) PRIMARY KEY NOT NULL AUTO_INCREMENT,
    emp_ID int(11),
    role_ID int(11),
    CONSTRAINT certification_employee_emp_ID_fk FOREIGN KEY (emp_ID) REFERENCES employee (emp_ID),
    CONSTRAINT certifications_role_role_ID_fk FOREIGN KEY (role_ID) REFERENCES role (role_ID)
);
CREATE UNIQUE INDEX certifications_cert_ID_uindex ON cmpsc431.certification (cert_ID);
CREATE UNIQUE INDEX certifications_employee_EID_fk ON cmpsc431.certification (emp_ID);
CREATE INDEX certifications_role_role_ID_fk ON cmpsc431.certification (role_ID);


CREATE TABLE cmpsc431.shift
(
    shift_ID int(11) PRIMARY KEY NOT NULL AUTO_INCREMENT,
    emp_ID int(11) NOT NULL,
    dept_ID int(11) NOT NULL,
    time_ID int(11) NOT NULL,
    week_ID int(11) NOT NULL,
    day_ID int(11) NOT NULL,
    status_ID int(11),
    pay_modifier double,
    day date,
    CONSTRAINT shift_employee_EID_fk FOREIGN KEY (emp_ID) REFERENCES employee (emp_ID),
    CONSTRAINT shift_department_DID_fk FOREIGN KEY (dept_ID) REFERENCES department (dept_ID),
    CONSTRAINT shift_shift_times_STID_fk FOREIGN KEY (time_ID) REFERENCES shift_time (time_ID),
    CONSTRAINT shift_week_WID_fk FOREIGN KEY (week_ID) REFERENCES week (week_ID),
    CONSTRAINT shift_weekday_WDID_fk FOREIGN KEY (day_ID) REFERENCES weekday (day_ID),
    CONSTRAINT shift_shift_status_SSID_fk FOREIGN KEY (status_ID) REFERENCES shift_status (status_ID)
);
CREATE INDEX shift_department_DID_fk ON cmpsc431.shift (dept_ID);
CREATE INDEX shift_employee_EID_fk ON cmpsc431.shift (emp_ID);
CREATE INDEX shift_shift_status_SSID_fk ON cmpsc431.shift (status_ID);
CREATE INDEX shift_shift_times_STID_fk ON cmpsc431.shift (time_ID);
CREATE INDEX shift_weekday_WDID_fk ON cmpsc431.shift (day_ID);
CREATE INDEX shift_week_WID_fk ON cmpsc431.shift (week_ID);

CREATE TABLE cmpsc431.department_need
(
    need_ID int(11) PRIMARY KEY NOT NULL AUTO_INCREMENT,
    week_ID int(11),
    day_ID int(11),
    time_ID int(11),
    dept_ID int(11),
    role_ID int(11),
    need int(11),
    day date,
    CONSTRAINT department_needs_week_week_ID_fk FOREIGN KEY (week_ID) REFERENCES week (week_ID),
    CONSTRAINT department_need_weekday_day_ID_fk FOREIGN KEY (day_ID) REFERENCES weekday (day_ID),
    CONSTRAINT department_need_shift_time_time_ID_fk FOREIGN KEY (time_ID) REFERENCES shift_time (time_ID),
    CONSTRAINT department_needs_department_dept_ID_fk FOREIGN KEY (dept_ID) REFERENCES department (dept_ID),
    CONSTRAINT department_needs_roles_role_ID_fk FOREIGN KEY (role_ID) REFERENCES role (role_ID)
);
CREATE INDEX department_needs_department_dept_ID_fk ON cmpsc431.department_need (dept_ID);
CREATE UNIQUE INDEX department_needs_need_ID_uindex ON cmpsc431.department_need (need_ID);
CREATE INDEX department_needs_roles_role_ID_fk ON cmpsc431.department_need (role_ID);
CREATE INDEX department_needs_week_week_ID_fk ON cmpsc431.department_need (week_ID);
CREATE INDEX department_need_shift_time_time_ID_fk ON cmpsc431.department_need (time_ID);
CREATE INDEX department_need_weekday_day_ID_fk ON cmpsc431.department_need (day_ID);


INSERT INTO cmpsc431.week (start_date, end_date) VALUES
('2017-10-01', '2017-10-07'),
('2017-10-08', '2017-10-14'),
('2017-10-15', '2017-10-21'),
('2017-10-22', '2017-10-28'),
('2017-10-29', '2017-11-04'),
('2017-11-05', '2017-11-11'),
('2017-11-12', '2017-11-18'),
('2017-11-19', '2017-11-25'),
('2017-11-26', '2017-12-02'),
('2017-12-03', '2017-12-09'),
('2017-12-10', '2017-12-16'),
('2017-12-17', '2017-12-23');

INSERT INTO cmpsc431.weekday (day_name) VALUES
('Sunday'),('Monday'),('Tuesday'),('Wednesday'),('Thursday'),('Friday'),('Saturday');

INSERT INTO cmpsc431.shift_status (status) VALUES ('Called in'),('Called off'),('Requested shift'),('Requested off');

INSERT INTO cmpsc431.shift_time (shift_start, shift_end) VALUES ('07:00:00', '15:00:00'),('15:00:00', '23:00:00'),('23:00:00', '07:00:00'),('07:00:00', '19:00:00'),('19:00:00', '07:00:00');

INSERT INTO cmpsc431.role (role) VALUES
('Registered Nurse'),('Licensed Practical Nurse'),('Nurse Practitioner'),
('Clinical Nurse Specialist'),('Nurse Assistant');

INSERT INTO cmpsc431.department (min_staff, max_staff, beds, dept_name) VALUES
(1, 10, 26, 'Emergency Room'),
(2, 6, 19, 'Intensive Care Unit'),
(1, 10, 12, 'Maternity'),
(1, 7, 25, 'Operating Room'),
(1, 9, 26, 'Quarantine'),
(2, 9, 25, 'Psychiatric Ward');

INSERT INTO cmpsc431.employee (home_dept, fname, mname, lname, ssn, phone1, phone2, start_date, end_date, full_time, salaried, pay_rate) VALUES
(null, 'Adam', 'Adam', 'Apple', '000-00-0001', '000-000-0001', '000-000-0001', '2017-10-11 01:00:46', null, 1, 1, 22.5),
(null, 'Brad', 'Brad', 'Baker', '000-00-0002', '000-000-0002', '000-000-0002', '2017-10-11 01:00:46', null, 0, 0, 15),
(null, 'Charles', 'Charles', 'Chaplan', '000-00-0003', '000-000-0003', '000-000-0003', '2017-10-11 01:00:46', null, 0, 1, 37),
(null, 'Derek', 'Derek', 'Davis', '000-00-0004', '000-000-0004', '000-000-0004', '2017-10-11 01:00:46', null, 1, 1, 40),
(null, 'Evan', 'Evan', 'Elliott', '000-00-0005', '000-000-0005', '000-000-0005', '2017-10-11 01:00:46', null, 1, 1, 21),
(null, 'Frank', 'Frank', 'Farris', '000-00-0006', '000-000-0006', '000-000-0006', '2017-10-11 01:00:46', null, 1, 1, 32),
(null, 'George', 'George', 'Grant', '000-00-0007', '000-000-0007', '000-000-0007', '2017-10-11 01:00:46', null, 0, 0, 19),
(null, 'Hank', 'Hank', 'Hamill', '000-00-0008', '000-000-0008', '000-000-0008', '2017-10-11 01:00:46', null, 0, 0, 21),
(null, 'Ivan', 'Ivan', 'Ikarov', '000-00-0009', '000-000-0009', '000-000-0009', '2017-10-11 01:00:46', null, 1, 0, 24),
(null, 'Jack', 'Jack', 'Joplin', '000-00-0010', '000-000-0010', '000-000-0010', '2017-10-11 01:00:46', null, 1, 1, 21),
(null, 'Kevin', 'Kevin', 'Keller', '000-00-0011', '000-000-0011', '000-000-0011', '2017-10-11 01:00:46', null, 1, 1, 20),
(null, 'Lenny', 'Lenny', 'Landman', '000-00-0012', '000-000-0012', '000-000-0012', '2017-10-11 01:00:46', null, 1, 1, 17),
(null, 'Mark', 'Mark', 'Morris', '000-00-0013', '000-000-0013', '000-000-0013', '2017-10-11 01:00:46', null, 1, 1, 32),
(null, 'Nick', 'Nick', 'Norton', '000-00-0014', '000-000-0014', '000-000-0014', '2017-10-11 01:00:46', null, 0, 1, 23),
(null, 'Orval', 'Orval', 'Obrian', '000-00-0015', '000-000-0015', '000-000-0015', '2017-10-11 01:00:46', null, 0, 0, 30),
(null, 'Peter', 'Peter', 'Parker', '000-00-0016', '000-000-0016', '000-000-0016', '2017-10-11 01:00:46', null, 0, 1, 39),
(null, 'Quinn', 'Quinn', 'Quarrick', '000-00-0017', '000-000-0017', '000-000-0017', '2017-10-11 01:00:46', null, 1, 0, 15),
(null, 'Robert', 'Robert', 'Rodgers', '000-00-0018', '000-000-0018', '000-000-0018', '2017-10-11 01:00:46', null, 1, 1, 17),
(null, 'Sam', 'Sam', 'Saville', '000-00-0019', '000-000-0019', '000-000-0019', '2017-10-11 01:00:46', null, 0, 1, 38),
(null, 'Tom', 'Tom', 'Tarantino', '000-00-0020', '000-000-0020', '000-000-0020', '2017-10-11 01:00:46', null, 0, 0, 27);
INSERT INTO cmpsc431.address (emp_ID, street1, street2, city, state, zip) VALUES
(1, '123 Easy Street', 'Apt. A', 'State College', 'Pennsylvania', '16801'),
(2, '124 Easy Street', 'Apt. B', 'State College', 'Pennsylvania', '16801'),
(3, '125 Easy Street', 'Apt. C', 'State College', 'Pennsylvania', '16801'),
(4, '126 Easy Street', 'Apt. D', 'State College', 'Pennsylvania', '16801'),
(5, '127 Easy Street', 'Apt. E', 'State College', 'Pennsylvania', '16801'),
(6, '128 Easy Street', 'Apt. F', 'State College', 'Pennsylvania', '16801'),
(7, '129 Easy Street', 'Apt. G', 'State College', 'Pennsylvania', '16801'),
(8, '130 Easy Street', 'Apt. H', 'State College', 'Pennsylvania', '16801'),
(9, '131 Easy Street', 'Apt. I', 'State College', 'Pennsylvania', '16801'),
(10, '132 Easy Street', 'Apt. J', 'State College', 'Pennsylvania', '16801'),
(11, '133 Easy Street', 'Apt. K', 'State College', 'Pennsylvania', '16801'),
(12, '134 Easy Street', 'Apt. L', 'State College', 'Pennsylvania', '16801'),
(13, '135 Easy Street', 'Apt. M', 'State College', 'Pennsylvania', '16801'),
(14, '136 Easy Street', 'Apt. N', 'State College', 'Pennsylvania', '16801'),
(15, '137 Easy Street', 'Apt. O', 'State College', 'Pennsylvania', '16801'),
(16, '138 Easy Street', 'Apt. P', 'State College', 'Pennsylvania', '16801'),
(17, '139 Easy Street', 'Apt. Q', 'State College', 'Pennsylvania', '16801'),
(18, '140 Easy Street', 'Apt. R', 'State College', 'Pennsylvania', '16801'),
(19, '141 Easy Street', 'Apt. S', 'State College', 'Pennsylvania', '16801'),
(20, '142 Easy Street', 'Apt. T', 'State College', 'Pennsylvania', '16801');

INSERT INTO cmpsc431.certification (emp_ID, role_ID) VALUES (1, 3),(2, 2),(3, 5),(4, 1),(5, 1),(6, 4),(7, 5),
(8, 1),(9, 5),(10, 2),(11, 4),(12, 3),(13, 4),(14, 2),(15, 4),(16, 1),(17, 4),(18, 5),(19, 2),(20, 4);
