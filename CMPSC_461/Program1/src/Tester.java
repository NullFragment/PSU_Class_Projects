import java.util.ArrayList;

public class Tester
{
    public static void main (String args[])
    {
        Lexer test1 = new Lexer("<body><b>blah</b></body>$");
        ArrayList<Token> tokens1 = new ArrayList<>();
        tokens1 = test1.GetTokens();

        Lexer test2 = new Lexer("<body> <b> blah </b> </body>  $");
        ArrayList<Token> tokens2 = new ArrayList<>();
        tokens2 = test2.GetTokens();

        Lexer test3 = new Lexer("<body> test# <b> blah </b> </body>  $");
        ArrayList<Token> tokens3 = new ArrayList<>();
        tokens3 = test3.GetTokens();


        System.out.println("Stop Here Plz");
    }
}
