package com.example.test1;

public class Words {
	
	public static void createWords(Board newBoard){
		if(newBoard.coordinates.size() == 0)
			return;
		if(Rules.rowFlag){
			createRowWord(newBoard, newBoard.coordinates.get(0));
			for(int i=0; i < newBoard.coordinates.size(); i++){
				createColWord(newBoard, newBoard.coordinates.get(i));
			}			
		}
		if(Rules.colFlag){
			createColWord(newBoard, newBoard.coordinates.get(0));	
			for(int i=0; i < newBoard.coordinates.size(); i++){
				createRowWord(newBoard, newBoard.coordinates.get(i));
			}		
		}
	}
	
	private static void createRowWord(Board newBoard, Point anchor){
		if(newBoard.board[anchor.row][anchor.col] == '0')
			return;
		else{
			String newWord = "";
			int seqStart = anchor.col;
			int seqEnd = anchor.col;
			while(seqStart >= 0 && newBoard.board[anchor.row][seqStart] != '0')
				seqStart--; //finds where the word begins
			while(seqEnd < Board.bWidth && newBoard.board[anchor.row][seqEnd] != '0')
				seqEnd++;  //finds where the word ends
			if(seqStart+1 != seqEnd-1){ //if the sequence is longer than 1 letter
				if(Board.left2right){
					for(int i=seqStart+1; i<seqEnd; i++)
						newWord += newBoard.board[anchor.row][i];
					newBoard.words.add(newWord);
				}
				else{
					for(int i=seqEnd-1; i>seqStart; i--)
						newWord += newBoard.board[anchor.row][i];
					newBoard.words.add(newWord);
				}
			}
		}
	}
	
	private static void createColWord(Board newBoard, Point anchor){
		if(newBoard.board[anchor.row][anchor.col] != '0'){
			String newWord = "";
			int seqStart = anchor.row;
			int seqEnd = anchor.row;
			while(seqStart >= 0 && newBoard.board[seqStart][anchor.col] != '0')
				seqStart--; //finds where the word begins
			while(seqEnd < Board.bHeight && newBoard.board[seqEnd][anchor.col] != '0')
				seqEnd++;  //finds where the word ends
			if(seqStart+1 != seqEnd-1){ //if the sequence is longer than 1 letter
				for(int i=seqStart+1; i<seqEnd; i++)
					newWord += newBoard.board[i][anchor.col];
				newBoard.words.add(newWord);
			}
		}
	}
	
}
