import java.util.ArrayList;

public class Lexer
{
    private final String letters = "abcdefghijklmnopqrstuvmxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    private final String digits = "0123456789";
    private final String tag_chars = "</>";

    private String statement;
    private int index = 0;
    private char current_ch;
    private ArrayList<Token> tokens = new ArrayList<>();

    public Lexer(String statement_in)
    {
        statement = statement_in;
        index = 0;
        current_ch = statement.charAt(index);
        FindTokens();
    }

    public ArrayList<Token> GetTokens()
    {
        return tokens;
    }


    private void FindTokens()
    {
        StringBuffer temp = new StringBuffer();
        boolean endOfInputReached = false;
        while (index <= statement.length() && !endOfInputReached)
        {
            temp.delete(0, temp.length());
            if (current_ch == '<')
            {
                temp = NextTokenString(letters + digits + tag_chars);
                tokens.add(new Token(temp.toString()));
            }
            else if (Character.isDigit(current_ch) || Character.isAlphabetic(current_ch))
            {
                temp = NextTokenString(letters + digits);
                tokens.add(new Token(temp.toString()));
            }
            else if (current_ch == '$')
            {
                tokens.add(new Token("$"));
                endOfInputReached = true;
            }
            else if(Character.isWhitespace(current_ch))
            {
                getNextChar();
            }
            else
            {
                Token test = new Token(Character.toString(current_ch));
                tokens.add(new Token(Character.toString(current_ch)));
                getNextChar();
            }
        }
    }

    private void getNextChar()
    {
        index++;
        current_ch = statement.charAt(index);
    }

    private StringBuffer NextTokenString(String toMatch)
    {
        StringBuffer nextString = new StringBuffer();
        while (toMatch.indexOf(current_ch) >= 0 && current_ch != '>')
        {
            nextString.append(current_ch);
            getNextChar();
        }
        if(current_ch == '>')
        {
            nextString.append(current_ch);
            getNextChar();
        }
        return nextString;
    }

}
