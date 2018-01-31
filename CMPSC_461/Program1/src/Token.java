public class Token
{
    public enum TokenType
    {
        openBody, closeBody, openBold, closeBold, openItalic, closeItalic, openList, closeList, openItem, closeItem,
        string, digit, invalid
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
            case digit:
            case string:
                return value;
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
            case digit:
                return "digit";
            case string:
                return "string";
            case invalid:
            default:
                return "Invalid Token";
        }
    }


}
