//
//  board.cpp
//  Final Project
//
//  Created by Michael DeLeo on 4/8/17.
//  Copyright © 2017 Michael DeLeo. All rights reserved.
//

#include "board.h"

object::object()
{
    
    bunny_win = false;
    martian_win = false;
    taz_win = false;
    tweety_win = false;
    
    std::random_device rd;
    
    int pieces [7];
    
    for (int i = 0; i < 7; i++)
    {
        pieces [i] = rd() % 25;
    }
    
    for (int i = 0; i < 7; i ++)
    {
        for (int j = i-1; j >=0 ; j--)
        {
            if (pieces[i] == pieces[j])
            {
                j = i;
                pieces[i] = rd() % 25;
            }
        }
    }
    
    tweety = "T";
    martian = "M";
    bunny = "B";
    taz = "D";
    //Makes sure the starting values are not the same
    
    int* temp;
    temp = pieces;
    setBoardSpace(temp);
    
    
    
}

void object::setBoardSpace(int* list)
{
    std::string temp[25];
    for (int i = 0; i < 25; i++)
    {
        temp[i] = " ";
    }
    
    temp[list[0]] = "M";
    temp[list[1]] = "B";
    temp[list[2]] = "D"; //Taz
    temp[list[3]] = "T";
    temp[list[4]] = "C";
    temp[list[5]] = "C";
    temp[list[6]] = "F";
         
         
    int counter = 0;
    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            board[i][j] = temp[counter];
            counter++;
            
        }
    }
    
    return;
}

void object::runSimulation()
{
    std::random_device rd;
    
    if (stillAround(bunny))
    {
        if (bunny == "B" && stillAround("C"))
        {
            setProtagonist(bunny, pathFinder("C", bunny), "C");
        }
        //Finding a carrot
        else if (bunny == "B(C)")
        {
            setProtagonist(bunny, pathFinder("F", bunny), "F");
        }
        //Finding the mountain
        else
        {
        setProtagonist(bunny, rd()%9, " ");
        }
        //No carrots left
    }
    
    if (stillAround(taz))
    {
        if (taz == "D" && stillAround("C"))
        {
            setProtagonist(taz, pathFinder("C", taz), "T");
        }
        //Finding a carrot
        else if (taz == "D(C)")
        {
            setProtagonist(taz, pathFinder("F", taz), "F");
        }
        //Finding the mountain
        else
        {
            setProtagonist(taz, rd()%9, " ");
        }
        //No carrots left
    }
    
    if (stillAround(tweety))
    {
        if (tweety == "T" && stillAround("C"))
        {
            setProtagonist(tweety, pathFinder("C", tweety),"C");
        }
        //Finding a carrot
        else if (tweety == "T(C)")
        {
            setProtagonist(tweety, pathFinder("F", tweety),"F");
        }
        //Finding the mountain
        else
        {
            setProtagonist(tweety, rd()%9, " ");
        }
        //No carrots left
    }
    
    
    if (martian == "M" && stillAround("C"))
    {
        setMartian(pathFinder("C", martian), "C");
    }
    //Finding a carrot
    else if (martian == "M(C)")
    {
        setMartian(pathFinder("F", martian), "F");
    }
    //Finding the mountain
    else
    {
        if (bunny == "B(C)")
        {
        setMartian(pathFinder(bunny,martian),bunny);
        }
        else if (taz == "D(C)")
        {
            setMartian(pathFinder(taz,martian),taz);
        }
        else if (tweety == "T(C)")
        {
            setMartian(pathFinder(tweety, martian),tweety);
        }
        else{
            setMartian(rd() % 9, " ");
        }
    }
    //No carrots left
    
    

    printBoard();
    
    //Analyzes board
    
    //TODO MIKE
    
    return;
}

bool object::gameOverYet()
{
    bool carrot_checker = true;
    bool character_checker = true;
    
    std::string character_array [3] = {bunny, taz, tweety};
    
    for (int i = 0; i <5; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                if (board[i][j] == "C" || character_array[k] == "B(C)" || character_array[k] == "T(C)" || character_array[k] == "D(C)")
                {
                    carrot_checker = false;
                }

            }
            for (int k = 0; k < 3; k++)
            {
                if (board[i][j] == character_array[k])
                {
                    character_checker = false;
                }
            }
            
        }
    }
    
    if (carrot_checker || character_checker)
    {
        //Game is Over
        return true;
    }
    else{
        //Game is not over
        return false;
    }
    
}

bool object::getResult()
{
    
    bool carrot_checker = true;
    bool character_checker = true;
    
    std::string character_array [3] = {bunny, taz, tweety};
    
    for (int i = 0; i <5; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            for (int k = 0; k < 3; k++ )
            {
                if (board[i][j] == character_array[k])
                {
                    character_checker = false;
                }
            }
        }
    }
    if (bunny_win)
    {
        std::cout << "Bunny wins!" << std::endl;
    }
    if (taz_win)
    {
        std::cout << "Tax wins!" << std::endl;
    }
    if (tweety_win)
    {
        std::cout << "Tweety wins!" << std::endl;
    }
    if(martian_win)
    {
        std::cout << "Martian wins!" << std::endl;
    }
    
    
    
    if (character_checker)
    {
        return false;
    }
    
    else{
        //getResult has been called too early
        std::cout << "ERROR IN getResult()" << std::endl;
        return false;
    }
    //Placeholder for return value
}

void object::printBoard()
{
    /*Example:
     
     _ _ _ _
     |B X X X |
     |X R X X |
     |X X G X |
     |G X X X |
     - - - -
     For the user, the Bomb and the two golds should be hidden until stepped on
     Spaces stepped on by the robot turn into "O"
     
     */
    
    std::cout << " _    _    _    _    _" << std::endl;
    for (int i = 0; i < 5; i++)
    {
        std::cout << "|";
        for (int j = 0; j < 5; j++)
        {
            /*
             if (board[i][j] == "B" || board[i][j] == "G" || board[i][j] == "X")
             {
             std::cout << "X ";
             }
             
             
             else
             */
            //For Testing
            std::cout << board[i][j] << "    ";
        }
        std::cout << "|" << std::endl;
    }
    std::cout << " -    -    -    -    -" << std::endl;
    
}



void object::setProtagonist(std::string & character, int move, std::string searching)
{
    if (stillAround(character))
    {}
    else{
        return;
    }
    int x_direc = 0;
    //-1 up | 1 down | 0 x is constant
    int y_direc = 0;
    //-1 left | 1 right | 0 y is constant
    if (move == 0)
    {
        x_direc = 0;
        y_direc = 0;
    }
    else if (move == 1)
    {
        y_direc = -1;
    }
    else if (move == 2)
    {
        y_direc = -1;
        x_direc = 1;
    }
    else if (move == 3)
    {
        x_direc = 1;
    }
    else if (move == 4)
    {
        x_direc = 1;
        y_direc = 1;
    }
    else if (move == 5)
    {
        y_direc = -1;
    }
    
    else if (move == 6)
    {
        x_direc = -1;
        y_direc = 1;
    }
    else if (move == 7)
    {
        x_direc = -1;
    }
    else if (move == 8)
    {
        x_direc = -1;
        y_direc = -1;
    }
    
    
    std::pair<int,int> current_loc = getPiece(character);
    
    bool check_for_carrot = false;
    for (int i = 0; i < character.size(); i++)
    {
        if (character[i] == 'C')
        {
            check_for_carrot = true;
        }
    }
    //Checks if the character has a carrot

    
    board[current_loc.first][current_loc.second] = " ";
    
    if (current_loc.first + y_direc < 0 || current_loc.first + y_direc > 4
        || current_loc.second +x_direc < 0 || current_loc.second + x_direc > 4)
    {
        
        //std::random_device rd;
        //board[current_loc.first] [current_loc.second] = character;
        int temp = pathFinder(" ", character);
        board[current_loc.first][current_loc.second] = character;
        std::cout << "ERROR IN setProtagonist() : " << character << std::endl;
        setProtagonist(character, temp, searching);
        
        
    }
    
    else if(board[current_loc.first + y_direc][current_loc.second + x_direc] == " ")
    {
        board[current_loc.first + y_direc][current_loc.second + x_direc] = character;
    }
    
    else if (board[current_loc.first + y_direc][current_loc.second + x_direc] == "C" && check_for_carrot)
    {
        std::random_device rd;
        board[current_loc.first] [current_loc.second] = character;
        int temp = pathFinder("F", character);
        setProtagonist(character, temp, "F");
        
    }
    
    else if (board[current_loc.first + y_direc][current_loc.second + x_direc] == "C")
    {
        board[current_loc.first + y_direc][current_loc.second + x_direc] = character + "(C)";
        character = character + "(C)";
    }
    
    else if (board[current_loc.first + y_direc][current_loc.second + x_direc] == martian)
    {
        if (martian == "M(C)" && check_for_carrot)
        {
            board[current_loc.first][current_loc.second] = "C";
            board[current_loc.first + y_direc][current_loc.second + x_direc] = martian;
        }
        if (martian == "M" && check_for_carrot)
        {
            martian = martian + "(C)";
            board[current_loc.first + y_direc][current_loc.second + x_direc] = martian;
        }
        else{
            board[current_loc.first + y_direc][current_loc.second + x_direc] = martian;
        }
    }
    
    else if (board[current_loc.first + y_direc][current_loc.second + x_direc] == "F" && check_for_carrot)
    {
        if (character == "B(C)")
        {
            bunny_win = true;
        }
        
        else if (character == "D(C)")
        {
            taz_win = true;
        }
        
        else if (character == "T(C)")
        {
            tweety_win = true;
        }
    }
    
    else
    {
        setProtagonist(character, pathFinder(searching, character), searching);
    }

    
    
    //board[current_loc.first - y_direc * speed][current_loc.second - x_direc] = "R";
    
    return;
    //TODO
    
}


void object::setMartian(int move, std::string searching)
{
    if (stillAround(martian))
    {}
    else{
        return;
    }
    int x_direc = 0;
    //-1 left | 1 right | 0 x is constant
    int y_direc = 0;
    //-1 down | 1 up | 0 y is constant
    if (move == 0)
    {
        x_direc = 0;
        y_direc = 0;
    }
    else if (move == 1)
    {
        y_direc = -1;
    }
    else if (move == 2)
    {
        y_direc = -1;
        x_direc = 1;
    }
    else if (move == 3)
    {
        x_direc = 1;
    }
    else if (move == 4)
    {
        x_direc = 1;
        y_direc = 1;
    }
    else if (move == 5)
    {
        y_direc = 1;
    }
    
    else if (move == 6)
    {
        x_direc = -1;
        y_direc = 1;
    }
    else if (move == 7)
    {
        x_direc = -1;
    }
    else if (move == 8)
    {
        x_direc = -1;
        y_direc = -1;
    }
    
    
    
    std::string character = martian;
    std::pair<int,int> current_loc = getPiece(character);
    
    bool check_for_carrot = false;
    for (int i = 0; i < character.size(); i++)
    {
        if (character[i] == 'C')
        {
            check_for_carrot = true;
        }
    }
    
    
    board[current_loc.first][current_loc.second] = " ";
    
    if (current_loc.first + y_direc < 0 || current_loc.first + y_direc > 4
        || current_loc.second + x_direc < 0 || current_loc.second + x_direc > 4)
    {
        //std::random_device rd;
       // board[current_loc.first] [current_loc.second] = character;
        int temp = pathFinder(" ", character);
        board[current_loc.first][current_loc.second] = martian;
        std::cout << "ERROR IN setMartian()" << std::endl;
        setMartian(temp,searching);
        
        
    }
    //Out of bounds
    
    else if(board[current_loc.first + y_direc][current_loc.second + x_direc] == " ")
    {
        board[current_loc.first + y_direc][current_loc.second + x_direc] = character;
    }
    
    
    else if (board[current_loc.first + y_direc][current_loc.second + x_direc] == "C" && check_for_carrot)
    {
        board[current_loc.first] [current_loc.second] = character;
        int temp = pathFinder("F", character);
        
        setMartian(temp, "F");
        
    }
    
    else if (board[current_loc.first + y_direc][current_loc.second + x_direc] == "C")
    {
        board[current_loc.first + y_direc][current_loc.second + x_direc] = character + "(C)";
        character = character + "(C)";
    }
    
    else if (board[current_loc.first + y_direc][current_loc.second + x_direc] == "F")
    {
        martian_win = true;
    }
    /*
    else if (check_for_carrot
        std::string char_array[3] = {bunny,taz,tweety};
        for (int i = 0; i < 3; i++)
        {
            if (char_array[i] == "B(C)" || char_array[i] == "T(C)" || char_array[i] == "D(C)")
            {
                martian =
            }
        }
    }
    */
    //check for all characters, if the character has a carrot, take it
    
    /*
    else{
        board[current_loc.first + y_direc][current_loc.second + x_direc] = "M";
    }
    */
        //board[current_loc.first - y_direc * speed][current_loc.second - x_direc] = "R";
        
        return;
    //TODO
    
}

void object::setMountain()
{
    std::random_device rd;
    
    int first = rd() % 5;
    int second = rd() % 5;
    
    while(board[first][second] != " ")
    {
        first = rd() % 5;
        second = rd() % 5;
    }
    
    std::pair<int,int> mount = getPiece(("F"));
    board[mount.first][mount.second] = " ";
    board[first][second] = "F";

    return;

}

std::pair<int,int> object::getPiece(std::string character)
{
    std::pair<int,int> temp;
    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            if (board[i][j] == character)
            {
                temp.first = i;
                temp.second = j;
                
                return temp;
            }
        }
    }
    printBoard();
    std::cout <<"ERROR in getPiece() : " << character << std::endl;
    if (character == "C" || character == "F")
    {}
    else{
        
    }
    return std::pair<int,int> (-1,-1);
    
    
}

bool object::stillAround(std::string character)
{
    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            if (board[i][j] == character)
            {
                return true;
            }
        }
    }
    
    return false;
}


int object::pathFinder(std::string searching, std::string character)
{
    //Ideas for fixing the bugs
    
    //Shrinking the subset to deal with a larger subset (creating a smaller subset and a "pseduo carrot"
    //Notching the subset to find the closest 'searching' piece
    //finish the part for decrypting the move
    
    
    std::random_device rd;
    
    std::pair<int,int> nothing;
    nothing.first = -1;
    nothing.second = -1;
    int move = 0;
    if (searching == " " || getPiece(searching) == nothing)
    {
        return rd() % 9;
    }
    if (getPiece(character) == nothing)
    {
        return 0;
    }
   
    std::pair<int,int> character_loc = getPiece(character);
    std::pair<int,int> finish_loc = getPiece(searching);
    int x = abs(character_loc.first - finish_loc.first) + 1;
    int y = abs(character_loc.second - finish_loc.second) + 1;

    
   
    
    //***********
	std::string ** sset = new std::string * [x];
	for (int p = 0; p < x; p++)
	{
		sset[p] = new std::string[y];
	}

    for (int i = 0; i < x; i++)
    {
        for (int j = 0; j < y; j++)
        {
            sset[i][j] = "X";
            std::cout << sset[i][j];
        }
        std::cout << std::endl;
    }
    //tests the size of the subset
    
    int c1 = abs(character_loc.first);
    int f1 = abs(finish_loc.first);
    
    int c2 = abs(character_loc.second);
    int f2 = abs(finish_loc.second);
    
	std::vector<std::string> flatBoard;
	for (int i = 0; i < 5; i++)
	{
		for (int j = 0; j < 5; j++)
		{
            if ( ((c1 <= i)&&(i <= f1)) || ((c1 >= i) &&(i >= f1)))
            {
                if (((c2 <= j) &&( j <= f2)) || ((c2 >= j) &&( j >= f2)))
                {
                    flatBoard.push_back(board[i][j]);
                    
                }
            }
		}
	}
    
    for (int i = 0; i < flatBoard.size(); i++)
    {
        std::cout << flatBoard.at(i);
    }
    
    std::cout << std::endl;
    
    std::pair<int,int> t;
    t.first = x;
    t.second = y;
    
    
	// initializes the flatboard
	int counter = 0;
	for (int i = 0; i < x; i++)
	{
		for (int j = 0; j < y; j++)
		{
			sset[i][j] = flatBoard.at(counter);
            counter++;
            std::cout << sset[i][j];
		}
        std::cout << std::endl;
    }
	//changes flatboard into subset
	
   // std::string ** subset;
    std::string ** subset = sset;
    /*
    if ((x > 3 && y > 3) && x*y > 9)
    {
        while ((x > 3 && y >3) && x*y > 9){
            std::pair<std::pair<int,int>, std::pair<int,int>> comb;
            comb.first = finish_loc;
            comb.second = character_loc;
            subSet = shrink(searching, t, comb ,sset);
    
            subset = subSet;
            x = x-1;
            y = y-1;
            flatBoard.clear();
        
            for (int i = 0; i < x; i++)
            {
                for (int j = 0; j < y; j++)
                {
                    flatBoard.push_back(subset[i][j]);
                }
            }
        }
    }
    else{
        subset = sset;
    }
    */
    for (int i = 0; i < x; i++)
    {
        for (int j = 0; j < y; j++)
        {
            std::cout << "X";
        }
        std::cout << std::endl;
    }
    
    

    /*
    int x_temp = 0;
    int y_temp = 0;
    std::string ** temp = new std::string * [x_temp];
    for (int i = 0; i < x_temp; i++)
    {
        temp[i] = new std::string [y_temp]
        
    }
    if (subset[0][0] == character)
    {
        
    }
    for (int i = 0; i < x; i++)
    {
        for (int j = 0; j < y; j++)
        {
            if (
            temp[i][j] = subset[i][j]
        }
    }
    */
    //Resizes the Board
    
	std::vector<int> flat_distance;
	
    int resistance;
    if (character == martian)
    {
        resistance = 1;
    }
    else{
        resistance = 1000;
    }
    
	for (int z = 0; z < x; z++)
	{
		for (int k = 0; k < y; k++)
		{
			for (int i = 0; i < x; i++)
			{
				for (int j = 0; j < y; j++)
				{
					if ((z == i && k == j)  && subset[i][j] == subset[z][k])
					{
						flat_distance.push_back(0);
					}
                    
                    else if ((subset[z][k] != character && subset[z][k] != searching && subset[z][k] != " ") || (subset[i][j] != character && subset[i][j] != searching && subset[i][j] != " "))
                    {
                        flat_distance.push_back(resistance);
                    }
					//checks that current element is both adjacent to Bugs (including diagonally) and empty for distance calculations
					else if ((subset[i][j] == " " || subset[i][j] == searching || subset[i][j] == character) && isNextTo(std::make_pair(z, k), std::make_pair(i, j)))
					{
						flat_distance.push_back(1);
					}

					else
					{
						flat_distance.push_back(1000);
					}
				}
			}
		}
	}


	//************************
	int ** distance = new int *[flatBoard.size()];
    for (int i = 0; i < flatBoard.size(); i++)
    {
        distance[i] = new int [flatBoard.size()];
    }
	int count = 0;
    for (int i = 0; i < flatBoard.size(); i++)
    {
        for (int j = 0; j < flatBoard.size(); j++)
        {
			distance[i][j] = flat_distance.at(count);
			count++;
            if (distance[i][j] == 1000)
            {
            std::cout << "Inf" << "    ";
            }
            else
            {
                std::cout << distance[i][j] << "    ";
            }
        }
        std::cout << std::endl;
    }
    
    
    //  A   B
    //A 0   x, y
    //B x,y 0
    std::vector<int> nodes;
    //Elements
    for (int i = 1; i <= flatBoard.size(); i++)
    {
        nodes.push_back(i);
    }
    //Array of the elements
    
    
     std::vector<std::vector<int>> unique;
    unique.push_back(nodes);
    //Looking for 2^(#nodes) vectors
    std::vector<std::vector<int>> copies;

    for (int i = 0; i < flatBoard.size(); i++)
    {
        do{
            std::vector<int> temp;
            temp.resize(nodes.size());
            std::copy(nodes.begin(),nodes.end(), temp.begin());
            
            for (int k = 0; k < i; k++)
            {
                temp.pop_back();
            }
            copies.push_back(temp);
        } while(next_combination(nodes.begin(),nodes.begin() +1, nodes.end()));
    }

        for (int k = 0; k < copies.size(); k++)
        {
            bool Store = true;//true until there is another vector that is the same
            int temp = -1;
            for (int j = 0; j < unique.size(); j++)
            {
                if (isVecEq(unique[j],copies[k]))
                {
                    Store = false;
                    temp = k;
                    
                }
                else{
                    temp = k;
                }
            }
            if (Store)
            {
                if (temp == -1)
                {}
                else{
                    unique.push_back(copies[temp]);
                }
            }
            //Push_back a copy vector if it is unique
        }
    //Makes all unique solutions
    
    int f_spot = 0;
    int Ch_spot = 0;
    for (int i = 0; i < flatBoard.size(); i++)
    {
        if (flatBoard[i] == searching)
        {
            f_spot = i+1;
        }
        
        if(flatBoard[i] == character)
        {
            Ch_spot = i+1;
        }
    }
    
    
    std::map<int, std::vector<int>> comb_distance;
    
    
    for(int z = 0; z < unique.size(); z++)
    {
        //Needs something that will let the loop iterate through every possibility 1-7 not just every permutation 1-7
        do
        {
            std::vector<int> copy = unique[z];
            
            int a = 0;;
            int b = 0;
            int newDistance = 0;
            for (int i = 1; i < copy.size(); i++)
            {
                a = copy[i-1]-1;
                b = copy[i] - 1;
                newDistance = newDistance + distance[a][b];
                // std::cout << "First, Second" << a << ", " << b << std::endl;
            }
            //To find the length of distance
            if(copy.back() == f_spot && copy.front() == Ch_spot)
            {
            comb_distance.insert(std::make_pair(newDistance, copy));
            }
            
                        // std::cout << newDistance << "-" << std::endl;
            // std::cout << elementDistance.back() << std::endl;
            
        }while(next_combination(unique[z].begin(),unique[z].begin() + 1, unique[z].end()));
        
        std::vector<int> copy = unique[z];
        
        int a = 0;;
        int b = 0;
        int newDistance = 0;
        for (int i = 1; i < copy.size(); i++)
        {
            a = copy[i-1]-1;
            b = copy[i] - 1;
            newDistance = newDistance + distance[a][b];
            
        }
        
        if (copy.back() == f_spot && copy.front() == Ch_spot)
        {
        comb_distance.insert(std::make_pair(newDistance, copy));
        }

    }
    
    
    
    
    
    //This creates the move from the character to the closest space
    std::pair<int,std::vector<int>> smallest = {1000,{1}};
    //size, location
    for (std::map<int,std::vector<int>>::iterator i = comb_distance.begin(); i != comb_distance.end(); i++)
    {
       if (smallest.first > i->first)
       {
           smallest.first = i->first;
           smallest.second = i->second;
       }
    }
    int first_position = (smallest.second).front();
    int second_position = (smallest.second).back();
    
    int difference_x = 0;
    int difference_y = 0;
    std::pair<int,int> node2;
    std::pair<int,int> node1;
    
    int ** node_mat = new int *[x];
    for (int i = 0; i < x; i++)
    {
        node_mat[i] = new int[y];
    }
    int coun = 0;
    for (int i = 0; i < x; i++)
    {
        for (int l = 0; l < y; l++)
        {
            node_mat[i][l] = nodes[coun];
            coun++;
            std::cout << node_mat[i][l] << " ";
        }
        std::cout << std::endl;
    }
    for (int i = 0; i < x; i++)
    {
        for (int j = 0; j < y; j++)
        {
            if (node_mat[i][j] == first_position)
            {
                node1.first = i;
                node1.second = j;
            }
            if (node_mat[i][j] == second_position)
            {
                node2.first = i;
                node2.second = j;
            }
        }
    }
    
    if (node2.first > node1.first)
    {
        difference_y = 1;
    }
    else if (node2.first < node1.first)
    {
        difference_y = -1;
    }
    else{}
    
    if (node2.second > node1.second)
    {
        difference_x = 1;
    }
    
    else if (node2.second < node1.second)
    {
        difference_x = -1;
    }
    else{}
    
	
	//Offset in subset

	//int difference_x = x_offset - character_loc.first;
	//int difference_y = y_offset - character_loc.second;

	
	for (int i = 0; i < flatBoard.size(); i++)
	{
		delete distance[i];
	}
	delete[] distance;

    
	for (int i = 0; i < x; i++)
	{
		//delete subset[i];
        delete node_mat[i];
	}
	delete[] sset;
    delete[] node_mat;
	//
    
    
    //-1 left | 1 right | 0 x is constant
    //-1 down | 1 up | 0 y is constant
    //For x_direc and y_direc in protagonist
    
    //8 1   2
    //7      3
    //6   5   4
    
    

    
    
	if (difference_x == 0 && difference_y == -1)
	{
		move = 1;
	}
	else if (difference_x == 1 && difference_y == 1)
	{
		move = 2;
	}
	else if (difference_x == 1 && difference_y == 0)
	{
		move = 3;
	}
	else if (difference_x == 1 && difference_y == -1)
	{
		move = 4;
	}
	else if (difference_x == 0 && difference_y == 1)
	{
		move = 5;
	}

	else if (difference_x == -1 && difference_y == 1)
	{
		move = 6;
	}
	else if (difference_x == -1 && difference_y == 0)
	{
		move = 7;
	}
	else if (difference_x == -1 && difference_y == -1)
	{
		move = 8;
	}
	//
    
    if (move == 0)
    {
        move = rd() % 9;
    }
    std::cout << character << ", move " << move << " to " << searching << std::endl;
    std::cout << difference_x << " " << difference_y << std::endl;
    return move;
}

/*
std::string ** object::resize(std::string searching, std::string character, std::pair<int,int> size)
{

}
*/


std::string ** object::shrink(std::string searching, std::pair<int,int> size, std::pair<std::pair<int,int>, std::pair<int,int>> bc, std::string ** set)
{
    std::pair<int,int> finalf = bc.first;
    std::pair<int,int> charc = bc.second;
    
    int carrotx = finalf.first;
    int carroty = finalf.second;
    
    int charcx = charc.first;
    int charcy = charc.second;
    
    int diffx = carrotx - charcx;
    int diffy = carroty - charcy;
    
    int newx = 0;
    int newy = 0;
    for (int i = 0; i < size.first; i++)
    {
        for (int j = 0; j < size.second; j++)
        {
            if (set[i][j] == searching)
            {
                newx = i;
                newy = j;
                i = size.first;
                j = size.second;
            }
        }
    }
    
    
    
    std::string ** newset = new std::string * [size.first - 1];
    for (int i = 0; i < (size.first - 1); i++)
    {
        newset[i] = new std::string [size.second - 1];
    }

    
    if (diffx > 0 && diffy > 0)
    {
        newx--;
        newy--;
        
        set[newx][newy] = searching;
        for (int i = 0; i < (size.first - 1); i++)
        {
            for (int j = 0; j < (size.second -1 ); j++)
            {
                newset[i][j] = set[i][j];
            }
        }
      
    }
    
    else if (diffx < 0 && diffy > 0)
    {
        newx++;
        newy--;
        
        set[newx][newy] = searching;
        for (int i = 1; i < (size.first); i++)
        {
            for (int j = 0; j < (size.second -1 ); j++)
            {
                newset[i-1][j] = set[i][j];
            }
        }
      
    
    }
    
    else if (diffx < 0 && diffy < 0)
    {
        newx++;
        newy++;
        
        set[newx][newy] = searching;
        for (int i = 0; i < (size.first - 1); i++)
        {
            for (int j = 0; j < (size.second -1 ); j++)
            {
                newset[i][j] = set[i+1][j+1];
            }
        }
       

    }
    
    else if (diffx > 0 && diffy < 0)
    {
        newx--;
        newy++;
        
        set[newx][newy] = searching;
        for (int i = 0; i < (size.first - 1); i++)
        {
            for (int j = 1; j < (size.second); j++)
            {
                newset[i][j-1] = set[i][j];
            }
        }

    }
    
    
    for (int i = 0 ; i < size.first -1; i++)
    {
        for (int j = 0; j < size.second -1; j++)
        {
            std::cout << newset[i][j];
        }
        std::cout << std::endl;
    }
    
    return newset;
    
}

bool object::isVecEq(std::vector<int> V1, std::vector<int> V2)
{
    if (V1.size() != V2.size())
    {
        return false;
    }
    int it_start = 0;
    for (int i = 0; i < V1.size();i++)
    {
        if (V1[i] == V2[i])
        {
            it_start++;
        }
        else{
            i = V1.size();
        }
    }
    //Starts the iterator position to save time
    
    if (it_start == V1.size())
    {
        return true;
    }
    //Checks if the entire vector is the same
   
    if (it_start != 0)
    {

        it_start = it_start -1;
    }
    bool trip = true;
    while(std::next_permutation(V2.begin() + it_start, V2.end()))
    {
        
        trip = true;
        for (int i = 0 + it_start; i < V1.size(); i++)
        {
            if (V1.at(i) != V2.at(i))
            {
                trip = false;
            }

        }
        if (trip)
        {
            return true;
        }
    }
    trip = true;
    for (int i = 0 + it_start; i < V1.size(); i++)
    {
        if (V1.at(i) != V2.at(i))
        {
            trip = false;
        }
        
    }
    if (trip)
    {
        return true;
    }
    
    return false;
}

bool object::isNextTo(std::pair<int,int> piece, std::pair<int,int> next)
{
	if (abs(next.first - piece.first) == 1 || abs(next.second - piece.second) == 1)
	{
		return true;
	}
	else
	{
		return false;
	}
}

template <typename Iterator>
inline bool object::next_combination(const Iterator first, Iterator k, const Iterator last)
{
    /* Credits: Thomas Draper */
    if ((first == last) || (first == k) || (last == k))
        return false;
    Iterator itr1 = first;
    Iterator itr2 = last;
    ++itr1;
    if (last == itr1)
        return false;
    itr1 = last;
    --itr1;
    itr1 = k;
    --itr2;
    while (first != itr1)
    {
        if (*--itr1 < *itr2)
        {
            Iterator j = k;
            while (!(*itr1 < *j)) ++j;
            std::iter_swap(itr1,j);
            ++itr1;
            ++j;
            itr2 = k;
            std::rotate(itr1,j,last);
            while (last != j)
            {
                ++j;
                ++itr2;
            }
            std::rotate(k,itr2,last);
            return true;
        }
    }
    std::rotate(first,k,last);
    return false;
}

































