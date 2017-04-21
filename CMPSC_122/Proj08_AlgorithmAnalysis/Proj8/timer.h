// timer.h for CMPSC122 lab 7
// Measure time in Window  
// Define a Timer object t, use t.start() for beginning of the algorithm, t.stop() for the ending, and t.show() for printing.

#ifndef TIMER_H
#define TIMER_H

#include <ctime>
#include <string>
#include <iostream>
using namespace std;

class Timer
{
  public:
    
    Timer();

    Timer(const std::string &label);

    ~Timer();

    void start(void);

    void stop(void);

    void show(void);
 
  private:

    void reset(void);

    std::string
      label;

    long 
      tps;

      clock_t 
        start_time,
        end_time;

    double 
      usertime,
      systemtime,
      elapsedtime,
      waittime;
};
#endif
// eof timer.h
// timer.cpp


Timer::Timer ()
{
  label = "Process Timer";
  reset();
}

Timer::Timer (const std::string &label)
{
  Timer::label = label;
  reset();
}

Timer::~Timer()
{
}

void
Timer::reset(void)
{
  tps = CLOCKS_PER_SEC;
  end_time = 0;
  usertime = 0;
  systemtime = 0;
  elapsedtime = 0;
  waittime = 0;
}

void
Timer::start(void)
{
    start_time = clock();
}

void
Timer::show(void)
{
 cout << "  "
      << label << "\n"
      << "  -------------------------------\n"
      << "  Elapsed Time   : "
      << elapsedtime
      << "s" << std::endl;

}

void
Timer::stop(void)
{
  end_time = clock();
  elapsedtime = ((double)(end_time -
                start_time )/(double)tps );
  if (elapsedtime < 0.001)
  {
    elapsedtime = 0.001;
  }

  if ( waittime < 0.00 )
  {
    waittime = 0.00;
  }
}

// eof timer.cpp
