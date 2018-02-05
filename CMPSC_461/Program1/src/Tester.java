/**
 * @author Kyle Salitrik
 * @PSU-ID kps168
 *
 * This class is simply used for running test cases.
 * Failing cases exit the code, so to test them, uncomment one at a time.
 */

import java.util.ArrayList;

public class Tester
{
    public static void main(String args[])
    {

        // Working test case
        Parser parse_test1 = new Parser("<body><b>blah</b></body>");
        parse_test1.run();

        // Working test case to test whitespace effects
        Parser parse_test2 = new Parser("<body> <b> blah </b> </body>  ");
        parse_test2.run();

        // Failing test case to test no <body>: Exit code -100
//        Parser parse_test3 = new Parser("<b>blah</body></b>");
//        parse_test3.run();

        // Failing test case to test out of order tags: Exit code -200
//        Parser parse_test4 = new Parser("<body><b>blah</body></b>");
//        parse_test4.run();

        // Failing test case to test invalid tokens: Exit code -300
//        Parser parse_test5 = new Parser("<body> test# <b> blah </b> </body>  ");
//        parse_test5.run();

        // Failing test case to test including a body within the body: Exit code -400
//        Parser parse_test6 = new Parser("<body> <body> <b> blah </b> </body></body>  ");
//        parse_test6.run();

        // Failing test case to test list item outside of list environment: Exit code -500
//        Parser parse_test7 = new Parser("<body><li><b>blah</b></li></body>");
//        parse_test7.run();


    }
}
