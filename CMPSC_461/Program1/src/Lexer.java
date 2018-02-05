/**
 * @author Kyle Salitrik
 * @PSU-ID kps168
 *
 * The Lexer class is responsible for breaking up the input statement in to token strings and then
 * creating an Array List of all tokens to be used by the Parser.
 */
import java.util.ArrayList;

public class Lexer
{
    // These final strings are used to be concatenated in order to search for valid token characters
    private final String letters = "abcdefghijklmnopqrstuvmxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    private final String digits = "0123456789";
    private final String tag_chars = "</>";

    private String statement;
    private int index = 0;
    private char current_ch;
    private ArrayList<Token> tokens = new ArrayList<>();

    /**
     * Constructor takes a statement and breaks it up into a token array list
     * @param statement_in - statement to be evaluated
     */
    Lexer(String statement_in)
    {
        statement = statement_in;
        index = 0;
        current_ch = statement.charAt(index);
        FindTokens();
    }

    /**
     * Simply returns the ArrayList of tokens to the calling class
     * @return - array list of all tokens from statement
     */
    public ArrayList<Token> GetTokens()
    {
        return tokens;
    }

    /**
     * This method tokenizes the input statement and adds each token individually to an ArrayList
     */
    private void FindTokens()
    {

        StringBuffer temp = new StringBuffer();
        boolean endOfInputReached = false;

        // Iterate over entire statement until statement is over or end of input is reached
        while (index <= statement.length() && !endOfInputReached)
        {
            temp.delete(0, temp.length());
            if (current_ch == '<')
            {
                // Checks if HTML tag is coming, if so, matches for tag characters and letters
                temp = NextTokenString(letters + tag_chars);
                tokens.add(new Token(temp.toString()));
            }
            else if (Character.isDigit(current_ch) || Character.isAlphabetic(current_ch))
            {
                // Checks if a text block is starting, then matches for digits or letters
                temp = NextTokenString(letters + digits);
                tokens.add(new Token(temp.toString()));
            }
            else if (current_ch == '$')
            {
                // Checks if EOI has been reached
                tokens.add(new Token("$"));
                endOfInputReached = true;
            }
            else if(Character.isWhitespace(current_ch))
            {
                // If character is whitespace, ignore it and keep iterating
                getNextChar();
            }
            else
            {
                // If no valid characters are found, create an invalid token
                Token test = new Token(Character.toString(current_ch));
                tokens.add(new Token(Character.toString(current_ch)));
                getNextChar();
            }
        }
    }

    /**
     * Simply advances the index and character string
     */
    private void getNextChar()
    {
        index++;
        current_ch = statement.charAt(index);
    }

    /**
     * This method gets the next potentially valid token string. It searches from the current character until either
     * a character does not match the given set of chars or the > character is reached.
     * @param toMatch - string of allowed characters for the token
     * @return - returns a string buffer of the token's value
     */
    private StringBuffer NextTokenString(String toMatch)
    {
        StringBuffer nextString = new StringBuffer();
        while (toMatch.indexOf(current_ch) >= 0 && current_ch != '>')
        {
            nextString.append(current_ch);
            getNextChar();
        }
        // Ensures the right carat is appended to HTML tags for proper identification
        if(current_ch == '>')
        {
            nextString.append(current_ch);
            getNextChar();
        }
        return nextString;
    }

}
