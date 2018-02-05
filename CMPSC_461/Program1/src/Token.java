/**
 * @author Kyle Salitrik
 * @PSU-ID kps168
 *
 * The token class is responsible for assigning the token's type as well as holding it's type and value information
 * for later retrieval.
 */

class Token
{
    // Simple enumeration of recognized tokens
    public enum TokenType
    {
        openBody, closeBody, openBold, closeBold, openItalic, closeItalic, openList, closeList, openItem, closeItem,
        string, invalid, EOI
    }

    /**
     * Constructor takes in the token's text value and then assigns it's type based on that input
     * @param input - token's value
     */
    Token(String input)
    {
        if(setType(input))
            value = input;
        else value = "Invalid Token: " + input; // If token is not valid, then make sure token is recognized as such
                                                // when value is printed out.
    }

    private TokenType type;
    private String value;

    /**
     * @return token type enumeration
     */
    TokenType getType()
    {
        return type;
    }

    /**
     * @return token's string value
     */
    String getValue()
    {
        return value;
    }

    /**
     * Returns the enumeration text value of a given token's tag type
     * @param givenTag - tag to match for text conversion
     * @return - string of tag type
     */
    String getTypeString(TokenType givenTag)
    {
        switch (givenTag)
        {
            case openBody:
                return "openBody";
            case closeBody:
                return "closeBody";
            case openBold:
                return "openBold";
            case closeBold:
                return "closeBold";
            case openItalic:
                return "openItalic";
            case closeItalic:
                return "closeItalic";
            case openList:
                return "openList";
            case closeList:
                return "closeList";
            case openItem:
                return "openItem";
            case closeItem:
                return "closeItem";
            case string:
                return "string";
            case EOI:
                return "EOI";
            case invalid:
            default:
                return "Invalid Token";
        }
    }

    /**
     * Overload for function. If no type is provided, it returns the type of the tag being referenced. Yay Java.
     * @return string of enumeration type
     */
    String getTypeString()
    {
        return getTypeString(type);
    }

    /**
     * This method sets the type of the token based on the given initialization string.
     * @param input - tokenized string from Lexer
     * @return - true if token is valid, false is invalid
     */
    private Boolean setType(String input)
    {
        // Checks if input string is potentially an HTML tag, then compares with switch-case
        if(input.startsWith("<"))
        {
            switch(input)
            {
                case "<body>":
                    type = TokenType.openBody;
                    return true;
                case "</body>":
                    type = TokenType.closeBody;
                    return true;
                case "<b>":
                    type = TokenType.openBold;
                    return true;
                case "</b>":
                    type = TokenType.closeBold;
                    return true;
                case "<i>":
                    type = TokenType.openItalic;
                    return true;
                case "</i>":
                    type = TokenType.closeItalic;
                    return true;
                case "<ul>":
                    type = TokenType.openList;
                    return true;
                case "</ul>":
                    type = TokenType.closeList;
                    return true;
                case "<li>":
                    type = TokenType.openItem;
                    return true;
                case "</li>":
                    type = TokenType.closeItem;
                    return true;
                default:
                    type = TokenType.invalid;
                    return false;
            }
        }

        // Checks if input is the start of a string
        else if(Character.isAlphabetic(input.charAt(0)) ||Character.isDigit(input.charAt(0))  )
        {
            String validText = "abcdefghijklmnopqrstuvmxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

            // Iterates over input string and verifies it follows string grammar rules.
            for (char character : input.toCharArray())
            {
                if (validText.indexOf(character) < 0)
                {
                    type = TokenType.invalid;
                    return false;
                }
            }
            type = TokenType.string;
            return true;
        }

        // Checks if input string is EOI character
        else if(input.charAt(0) == '$')
        {
            type = TokenType.EOI;
            return true;
        }

        // Otherwise sets token type to invalid
        else
        {
            type = TokenType.invalid;
            return false;
        }
    }



}
