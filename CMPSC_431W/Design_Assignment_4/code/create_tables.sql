CREATE TABLE employee
(
    EID INT(11) PRIMARY KEY NOT NULL AUTO_INCREMENT,
    home_dept INT(11),
    fname CHAR(50) NOT NULL,
    mname CHAR(50),
    lname CHAR(50),
    ssn CHAR(12),
    phone1 CHAR(13),
    phone2 CHAR(13),
    start_date DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
    end_date DATE,
    full_time TINYINT(1) DEFAULT '0' NOT NULL,
    salaried TINYINT(1) DEFAULT '0' NOT NULL,
    pay_rate DOUBLE DEFAULT '0' NOT NULL,
    CONSTRAINT employee_department_DID_fk FOREIGN KEY (home_dept) REFERENCES department (DID)
);
CREATE INDEX employee_department_DID_fk ON employee (home_dept);

CREATE TABLE department
(
    DID INT(11) PRIMARY KEY NOT NULL AUTO_INCREMENT,
    charge_nurse INT(11),
    min_staff INT(11) DEFAULT '0' NOT NULL,
    max_staff INT(11) DEFAULT '100' NOT NULL,
    beds INT(11) DEFAULT '0' NOT NULL,
    dept_name CHAR(50) NOT NULL,
    CONSTRAINT department_employee_EID_fk FOREIGN KEY (charge_nurse) REFERENCES employee (EID)
);
CREATE INDEX department_employee_EID_fk ON department (charge_nurse);

CREATE TABLE shift_status
(
    SSID INT(11) PRIMARY KEY NOT NULL AUTO_INCREMENT,
    status CHAR(50)
);

CREATE TABLE shift_times
(
    STID INT(11) PRIMARY KEY NOT NULL AUTO_INCREMENT,
    shift_start TIME NOT NULL,
    shift_end TIME NOT NULL
);

CREATE TABLE weekday
(
    WDID INT(11) PRIMARY KEY NOT NULL AUTO_INCREMENT,
    day_name CHAR(10)
);

CREATE TABLE week
(
    WID INT(11) PRIMARY KEY NOT NULL AUTO_INCREMENT,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL
);

CREATE TABLE shift
(
    SID INT(11) PRIMARY KEY NOT NULL AUTO_INCREMENT,
    employee INT(11) NOT NULL,
    department INT(11) NOT NULL,
    shift_time INT(11) NOT NULL,
    week_id INT(11) NOT NULL,
    dow INT(11) NOT NULL,
    shift_status INT(11),
    pay_modifier DOUBLE,
    CONSTRAINT shift_employee_EID_fk FOREIGN KEY (employee) REFERENCES employee (EID),
    CONSTRAINT shift_department_DID_fk FOREIGN KEY (department) REFERENCES department (DID),
    CONSTRAINT shift_shift_times_STID_fk FOREIGN KEY (shift_time) REFERENCES shift_times (STID),
    CONSTRAINT shift_week_WID_fk FOREIGN KEY (week_id) REFERENCES week (WID),
    CONSTRAINT shift_weekday_WDID_fk FOREIGN KEY (dow) REFERENCES weekday (WDID),
    CONSTRAINT shift_shift_status_SSID_fk FOREIGN KEY (shift_status) REFERENCES shift_status (SSID)
);
CREATE INDEX shift_department_DID_fk ON shift (department);
CREATE INDEX shift_employee_EID_fk ON shift (employee);
CREATE INDEX shift_shift_status_SSID_fk ON shift (shift_status);
CREATE INDEX shift_shift_times_STID_fk ON shift (shift_time);
CREATE INDEX shift_weekday_WDID_fk ON shift (dow);
CREATE INDEX shift_week_WID_fk ON shift (week_id);