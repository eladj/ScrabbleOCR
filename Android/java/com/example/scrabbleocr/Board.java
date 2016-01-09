package com.example.test1;

import java.util.Vector;

//Game board
public class Board {
	//Board characteristics  
	public static int bHeight = 9; //Board height.
	public static int bWidth = 9; // Board width.
	public static int playerLetters = 8; //Legal amount of letters per player.
	public static boolean firstMoveFlag = true;
	public static boolean left2right = true;
	
	public char[][] board = new char [bHeight][bWidth];
	Vector<Point> coordinates = new Vector<Point>(0, 1);
	Vector<String> words = new Vector<String>(0,1);
	Vector<Boolean> wordsLegality =  new Vector<Boolean>(0,1);
	
	
	public Board (){
		for(int i=0; i<bHeight; i++){
			for(int j=0; j<bWidth; j++){
				board[i][j]='0';
			}
		}		
	}
	
	// Copy constractor.
	public Board (Board other){
		for(int i=0; i<bHeight; i++){
			for(int j=0; j<bWidth; j++){
				this.board[i][j] = other.board[i][j];
			}
		}
	}	
	
	//Returns true if two boards are identical and false otherwise. Updates coordinates changes.
	public boolean sameBoard (Board other){
		boolean areTheSame = true;
		for(int row=0; row<bHeight; row++){
			for(int col=0; col<bWidth; col++){
				if(this.board[row][col] != other.board[row][col]){ 
					// adding coordinates of the differences to the new board (this board)
					this.coordinates.addElement(new Point(row,col));
					areTheSame = false;
				}
			}
		}
		return areTheSame;
	}

}
