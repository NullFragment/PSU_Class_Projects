/**
 * @author Kyle Salitrik
 * @PSU-ID kps168
 *
 * The parser class iterates through all of the tokens gathered by the Lexer and ensures
 * they follow the rules of the grammar. It also prints out the HTML structure
 * and notifies user of any encountered errors.
 *
 * WARNING: ERRORS EXIT THE PROGRAM!
 * I was not sure if this was the desired behavior. They are easy enough to comment out if necessary.
 * Error exits are on lines:
 *          48
 *          75
 *          88
 *         112
 *         123
 */

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Stack;

public class Parser
{
    private Lexer lexer;
    private ArrayList<Token> tokens;

    /**
     * This constructor takes in the statement to be evaluated and passes it along to the Lexer.
     * @param statement_in - statement to be parsed
     */
    public Parser(String statement_in)
    {
        lexer = new Lexer(statement_in + "$");
        tokens = lexer.GetTokens();
    }

    /**
     * This method runs the Lexer and Parser code to format the HTML output.
     */
    public void run()
    {
        // Checks if the statement begins with a <body> tag. If not, exits and reports failure.
        if (tokens.get(0).getType() != Token.TokenType.openBody)
        {
            System.err.println("ERROR: Website must start and end with proper body tags:");
            System.err.println("<body> ... </body>");
            System.exit(-100);
        }

        // If beginning is valid, continue checking the rest of the tags.
        else
        {
            // These two arraylists are used to quickly check if the tag opens or closes a section of the HTML body.

            ArrayList<Token.TokenType> openTags = new ArrayList<>(Arrays.asList(Token.TokenType.openBody,
                    Token.TokenType.openBold, Token.TokenType.openItalic, Token.TokenType.openItem,
                    Token.TokenType.openList));

            ArrayList<Token.TokenType> closeTags = new ArrayList<>(Arrays.asList(Token.TokenType.closeBody,
                    Token.TokenType.closeBold, Token.TokenType.closeItalic, Token.TokenType.closeItem,
                    Token.TokenType.closeList));

            int indent = 0; // Tracks indentation level
            Token currentToken; // Current token to be evaluated
            Stack<Token> tagHistory = new Stack<>();    // This stack is used to remember the last opening HTML tag
                                                        // used and provides a history

            // Runs token evaluation until an error is found or no tokens are left from the Lexer.
            while (!tokens.isEmpty())
            {
                // Get front of tokens from Lexer
                currentToken = tokens.remove(0); // Pops the front token from the array


                if(currentToken.getType() == Token.TokenType.openBody && indent > 0)
                {
                    System.err.println("A body cannot be placed within the body of a website");
                    System.exit(-400);
                }

                // Check if new environment is starting
                if (openTags.indexOf(currentToken.getType()) >= 0)
                {
                    if(currentToken.getType() == Token.TokenType.openItem
                            && tagHistory.peek().getType() != Token.TokenType.openList)
                    {
                        System.err.println("List item started outside of list environment.");
                        System.err.println("Last opened environment is: " +
                                currentToken.getTypeString(tagHistory.peek().getType()));
                        System.exit(-500);
                    }
                    tagHistory.push(currentToken);
                    PrintToken(indent, currentToken);
                    indent++;
                }

                // Checks if environment is closing
                else if (closeTags.indexOf(currentToken.getType()) >= 0)
                {
                    Token.TokenType validClosingToken = findMatchingTag(tagHistory.peek().getType());

                    // If closing token is valid, I.E. <b>...</b>, then continue
                    // If not, print an error and print expected token.
                    if (currentToken.getType() == validClosingToken)
                    {
                        indent--;
                        PrintToken(indent, currentToken);
                        tagHistory.pop();
                    }
                    else
                    {
                        System.err.println("Invalid Closing Tag: " + currentToken.getValue());
                        System.err.println("Expected Closing Tag: " + currentToken.getTypeString(validClosingToken));
                        System.exit(-200);
                    }
                }

                // Checks if invalid token is encountered
                else if(currentToken.getType() == Token.TokenType.invalid)
                {
                    System.err.println("Invalid token encountered.");
                    System.err.println(currentToken.getValue());
                    System.exit(-300);
                }

                // Checks if token is EOI, if not prints the string (only option left)
                else if(currentToken.getType() != Token.TokenType.EOI)
                {
                    PrintToken(indent, currentToken);
                }
            }
        }

    }

    /**
     * This method simply prints the current token's value out to the console
     * @param indentLevel - number of indentations to place
     * @param token - reference to token to print out
     */
    private void PrintToken(int indentLevel, Token token)
    {
        // Print indents
        for (int i = 0; i < indentLevel; i++) System.out.print("  ");

        // Print token value
        System.out.println(token.getValue());

    }

    /**
     * Finds the matching closing token type to a given opening token type. This information is obtained from the
     * token history stack.
     *
     * @param toMatch - token type to match with
     * @return - returns matching token type
     */
    private Token.TokenType findMatchingTag(Token.TokenType toMatch)
    {
        switch (toMatch)
        {
            case openBody:
                return Token.TokenType.closeBody;
            case openBold:
                return Token.TokenType.closeBold;
            case openItalic:
                return Token.TokenType.closeItalic;
            case openItem:
                return Token.TokenType.closeItem;
            case openList:
                return Token.TokenType.closeList;
            default:
                return Token.TokenType.invalid;
        }
    }
}
