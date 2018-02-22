#ifndef BOARD_H_INCLUDED
#define BOARD_H_INCLUDED

#include <vector>

class Board
{
private:
    std::vector<char> _board;
public:
	const static char INVALID_SQUARE;
	const static char EMPTY;
	const static char RED;
	const static char BLACK;
	const static char K_RED;
	const static char K_BLACK;

    Board();
	Board & UpdateKings();
    std::vector<char> & GetBoard();
    char & operator[](int index);
    void Print();
    std::vector<std::vector<char>> GetBoardAsMatrix();
	int getPieceCount(const char piece); 
};

#endif