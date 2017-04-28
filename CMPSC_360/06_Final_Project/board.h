//
//  board.h
//  Final Project
//
//  Created by Michael DeLeo on 4/8/17.
//  Copyright © 2017 Michael DeLeo. All rights reserved.
//

#ifndef board_h
#define board_h

#include <iostream>
#include <string>
#include <random>
#include <utility>
#include <algorithm>
#include <iterator>
#include <cmath>
#include <map>

class object
{
public:
    object();
    //Operator''
    
    
    //Order is bunny, taz, tweety, martian
    void setMartian(int move, std::string searching);
    //Precondition: move
    //Postcondition: sets the martian to a new space. Kills a character if its there and steals a carrot
    
    void setMountain();
    //Precondition: Nothing
    //Postcondition: Sets the mountain to a new unoccupied space
   
    void setProtagonist(std::string & character, int move, std::string searching);
    //Precondition: character's value, and its move
    //Postcondition: sets that protagonist (bugs, taz or tweety) to a new space
    
    void printBoard();
    //Precondition: Nothing | Postcondition: Prints Board
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
    
    void runSimulation();
    //Precondition: Nothing | Postcondition: Uses set functions to change the board one iteration
    
    bool getResult();
    //Precondition: Nothing | Postcondition: False Bomb wins, True Robot Wins
    
    bool gameOverYet();
    //Precondition: Nothing | Postcondition: True Game is over, False Game is still going
    
private:
    
    int pathFinder(std::string searching, std::string character);
    //Precondition: Object that the character is looking for, and character
    //Postcondition: the space that will bring the character closest

	bool isNextTo(std::pair<int, int> piece, std::pair<int, int> next);
	//Preconditionn: Coordinates of two pieces
	//Postcondition: True next to each, False  are not next to each other
    
    bool isVecEq(std::vector<int> V1, std::vector<int> V2);
    //Precondition:: Two vectors
    //Postcondition: For every permutation, if there is one case they are equal, then return true
    
    std::string ** resize(std::string searching, std::string character, std::pair<int,int> size);
    //Precondition:: character and searching varaibles, same as pathfinder. size (x,y), and the array
    //Postcondition: resizes the array until it finds the closest 'searching'
    
    std::string ** shrink(std::string searching, std::pair<int,int> size, std::pair<std::pair<int,int>,std::pair<int,int> > bc, std::string ** set);
    //Precondition: character and searching varaibles, same as pathfinder. size(x,y), and the array
    //Postcondition: shrinks the array until it is 3x3 with a pseudo 'searhing'
    std::string bunny;
    std::string martian;
    std::string taz;
    std::string tweety;
    
    bool bunny_win;
    bool martian_win;
    bool taz_win;
    bool tweety_win;
    
    bool stillAround(std::string character);
    //Precondition: character
    //Postcondition: True if character is alive, False if character is not
    
    void setBoardSpace(int* list);
    
    std::pair<int,int> getPiece(std::string character);
    //Precondition: Character's string value
    //Postcondition: returns the character's position
    
    std::string board[5][5];
    //R: Robot | G: Gold | B: Bomb | X: nothing
    
    template <typename Iterator>
    inline bool next_combination(const Iterator first, Iterator k, const Iterator last);
};

#endif /* board_h */
