package com.example.test1;
import android.app.Activity;
import android.content.Context;
import android.util.Log;

public class Game extends Activity{
	private static final String TAG = "myApp";

	public static void main(Context myContext) {
		Vocabulary vocabulary = new Vocabulary(myContext);
		Board oldBoard = new Board();
		Board newBoard = new Board();
		
		oldBoard.board[2][0] = 'p';
		oldBoard.board[2][1] = 'o';
		oldBoard.board[2][2] = 'l';
		oldBoard.board[2][3] = 'l';
		oldBoard.board[1][0] = 'u';
		oldBoard.board[3][0] = 'l';
		oldBoard.board[4][0] = 'o';
		oldBoard.board[5][0] = 'a';
		oldBoard.board[6][0] = 'd';
		
		
		newBoard = new Board(oldBoard);
		Board.firstMoveFlag = false;
		
		newBoard.board[0][1] = 's';
		newBoard.board[1][1] = 'p';
		newBoard.board[3][1] = 'o';
		newBoard.board[4][1] = 'n';
		
		for(int i=0; i<Board.bHeight; i++){
			for(int j=0; j<Board.bWidth; j++){
				System.out.print(newBoard.board[i][j]+" ");
			}
			System.out.print("\n");
		}
		
		if(newBoard.sameBoard(oldBoard))
			System.out.println("\nNo changes has been made");
		else{
			System.out.println("\nchanges has been made in " + newBoard.coordinates.size() + " places");
			if(Rules.validMove(oldBoard.board, newBoard, vocabulary)) {
				System.out.println("The move is legal");
				Log.v(TAG, "The move is legal");
			}
			else {
				System.out.println("The move is illegal");
				Log.v(TAG, "The move is illegal");
			}
			
		}
		
		String isLegal = "";
		for(int i=0; i<newBoard.words.size(); i++){
			if(newBoard.wordsLegality.get(i))
				isLegal = "Legal";
			else
				isLegal = "Illegal";
			System.out.println(newBoard.words.get(i) + " - " + isLegal); 
		}
		
		
	}
	

}
