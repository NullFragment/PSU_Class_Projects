class Token
{
    public enum TokenType
    {
        openBody, closeBody, openBold, closeBold, openItalic, closeItalic, openList, closeList, openItem, closeItem,
        string, invalid, EOI
    }

    Token(String input)
    {
        if(setType(input))
            value = input;
        else value = "Invalid Token: " + input;
    }

    private TokenType type;
    private String value;

    TokenType getType()
    {
        return type;
    }

    String getValue()
    {
        return value;
    }

    String getPrintString()
    {
        switch (type)
        {
            case openBody:
                return "<body>";
            case closeBody:
                return "</body>";
            case openBold:
                return "<b>";
            case closeBold:
                return "</b>";
            case openItalic:
                return "<i>";
            case closeItalic:
                return "</i>";
            case openList:
                return "<ul>";
            case closeList:
                return "</ul>";
            case openItem:
                return "<li>";
            case closeItem:
                return "</li>";
            case string:
                return value;
            case EOI:
                return "";
            case invalid:
            default:
                return "Invalid Token";
        }
    }

    String getTypeString()
    {
        switch (type)
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

    private Boolean setType(String input)
    {
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
                case "<lu>":
                    type = TokenType.openItem;
                    return true;
                case "</lu>":
                    type = TokenType.closeItem;
                    return true;
                default:
                    type = TokenType.invalid;
                    return false;
            }
        }
        else if(Character.isAlphabetic(input.charAt(0)))
        {
            String validText = "abcdefghijklmnopqrstuvmxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

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
        else if(input.charAt(0) == '$')
        {
            type = TokenType.EOI;
            return true;
        }

        else
        {
            type = TokenType.invalid;
            return false;
        }
    }



}
