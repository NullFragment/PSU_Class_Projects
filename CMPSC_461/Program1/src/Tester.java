import java.util.ArrayList;

public class Tester
{
    public static void main (String args[])
    {

//        Parser parse_test1 = new Parser("<body><b>blah</b></body>");
//        parse_test1.run();

//        Parser parse_test2 = new Parser("<body> <b> blah </b> </body>  ");
//        parse_test2.run();

//        Parser parse_test3 = new Parser("<body> test# <b> blah </b> </body>  ");
//        parse_test3.run();

//        Parser parse_test4 = new Parser("<body><b>blah</body></b>");
//        parse_test4.run();

//        Parser parse_test5 = new Parser("<b>blah</body></b>");
//        parse_test5.run();

        Parser parser =
                new Parser ("<body> google <b><i><b> yahoo</b></i></b></body>");
        parser.run();

    }
}
