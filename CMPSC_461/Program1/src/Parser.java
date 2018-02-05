import java.util.ArrayList;
import java.util.Arrays;
import java.util.Stack;

public class Parser
{
    private Lexer lexer;
    private ArrayList<Token> tokens;

    public Parser(String statement_in)
    {
        lexer = new Lexer(statement_in + "$");
        tokens = lexer.GetTokens();
    }

    public void run()
    {
        if (tokens.get(0).getType() != Token.TokenType.openBody)
        {
            System.err.println("ERROR: Website must start and end with proper body tags:");
            System.err.println("<body> ... </body>");
            System.exit(-100);
        }
        else
        {
            ArrayList<Token.TokenType> openTags = new ArrayList<>(Arrays.asList(Token.TokenType.openBody,
                    Token.TokenType.openBold, Token.TokenType.openItalic, Token.TokenType.openItem,
                    Token.TokenType.openList));

            ArrayList<Token.TokenType> closeTags = new ArrayList<>(Arrays.asList(Token.TokenType.closeBody,
                    Token.TokenType.closeBold, Token.TokenType.closeItalic, Token.TokenType.closeItem,
                    Token.TokenType.closeList));

            int indent = 0;
            Token currentToken;
            Stack<Token> tagHistory = new Stack<>();

            while (!tokens.isEmpty())
            {
                // Get front of tokens from Lexer
                currentToken = tokens.remove(0);

                // Check if environment is starting or ending
                if (openTags.indexOf(currentToken.getType()) >= 0)
                {
                    tagHistory.push(currentToken);
                    PrintToken(indent, currentToken);
                    indent++;
                }
                else if (closeTags.indexOf(currentToken.getType()) >= 0)
                {
                    Token.TokenType validClosingToken = findMatchingTag(tagHistory.peek().getType());
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
                else if(currentToken.getType() == Token.TokenType.invalid)
                {
                    System.err.println("Invalid token encountered.");
                    System.err.println(currentToken.getValue());
                    System.exit(-300);
                }
                else if(currentToken.getType() != Token.TokenType.EOI)
                {
                    PrintToken(indent, currentToken);
                }
            }
        }

    }

    private void PrintToken(int indentLevel, Token token)
    {
        // Print indents
        for (int i = 0; i < indentLevel; i++) System.out.print("  ");

        // Print token value
        System.out.println(token.getValue());

    }

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
