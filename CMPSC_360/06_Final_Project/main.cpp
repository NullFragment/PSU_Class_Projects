//
//  main.cpp
//  Final Project
//
//  Created by Michael DeLeo on 4/8/17.
//  Copyright © 2017 Michael DeLeo. All rights reserved.
//

#include "board.h"


int main()
{
    object new_board;
    
    new_board.printBoard();
    
    int counter = 0;
    while (!(new_board.gameOverYet()))
    {
        new_board.runSimulation();
        counter++;
        if (counter % 3 == 0)
        {
            new_board.setMountain();
        }
    }
    
    bool result = new_board.getResult();
    
    if (result)
    {
        std::cout << "Protaganist wins!" << std::endl;
        new_board.printBoard();
    }
    
    else
    {
        std::cout << "Marvin the Martian wins!" << std::endl;
    }
    
    return 0;
}
