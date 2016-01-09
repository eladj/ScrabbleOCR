package com.example.test1;

// checks if a game move is legal;
public class Rules {
	
	public static boolean rowFlag = false;
	public static boolean colFlag = false;
	
	//main function to check if a move is legal
	public static boolean validMove(char[][] oldBoard ,Board newBoard, Vocabulary vocabulary){
		rowFlag = false;
		colFlag = false;
		if(newBoard.coordinates.size() == 0)
			return true;
		if(newBoard.coordinates.size() > Board.playerLetters)
			return false;
		else{
			boolean wordsAreLegal = true;
			rowFlag = RowRules.validMove(oldBoard, newBoard.coordinates); 
			if(!(rowFlag))
				colFlag = ColRules.validMove(oldBoard, newBoard.coordinates);
			if(rowFlag || colFlag){
				Words.createWords(newBoard);
				vocabulary.areWordsLegal(newBoard.words, newBoard.wordsLegality);
				for(int i=0; i<newBoard.wordsLegality.size(); i++)
					if(newBoard.wordsLegality.get(i) == false)
						wordsAreLegal = false;
			}
			return((rowFlag || colFlag) && wordsAreLegal);
				
		}
	}	
}
	

	