package com.example.test1;

import java.util.Vector;

public class RowRules {
	
	//checks if all the new letters are at the same row and calls validSequenceMove if true.
	public static boolean validMove(char[][] oldBoard, Vector<Point> changes){
		if(changes.size() == 0)
			return true;
		for(int i=0, row = changes.firstElement().row; i<changes.size(); i++){
			if(changes.get(i).row != row)
				return false;
		}
		
		return validSequenceMove(oldBoard, changes);
	}
	
	//returns true if the new letters are in a legal sequence regarding old letters as well.
	private static boolean validSequenceMove(char[][] oldBoard, Vector<Point> changes){
		if(changes.size()==0)
			return true;
		Vector<Sequence> sequences = new Vector<Sequence>(0,1);
		createRowSequences(sequences, changes);
		
		System.out.println("there are " + sequences.size() + " sequences:");
		for(int i=0; i<sequences.size(); i++){
			System.out.println("	sequence from " + sequences.get(i).startingPoint.row 
					+ "," + sequences.get(i).startingPoint.col + " to " + sequences.get(i).endingPoint.row 
					+ "," + sequences.get(i).endingPoint.col);
		}
		
		// If this is the first move of the game
		if(Board.firstMoveFlag == true){
			if(sequences.size()==0)
				return false;
			if(sequences.size() == 1){
				Board.firstMoveFlag = false;
				return true;
			}
			else
				return false;
		}

		return SeqIsLegal(oldBoard, sequences);	
	}
	
	//creates row sequences of the new letters according to starting point and ending point.
	private static void createRowSequences(Vector<Sequence> sequences, Vector<Point> changes){
		int seqStart = changes.get(0).col; //the first sequence start at the first element's column.
		int row = changes.get(0).row;
		
		int i = 0;
		while(i<changes.size()-1){
			if(changes.get(i).col+1 != changes.get(i+1).col){
				sequences.addElement(new Sequence(new Point(row,seqStart), new Point(row, changes.get(i).col)));
				seqStart = changes.get(i+1).col;
			}
			i++;
		}
		sequences.addElement(new Sequence(new Point(row,seqStart), new Point(row, changes.get(i).col)));
	}
	
	//returns true if a given sequence is connected to at least one old letter.
	private static boolean oneSeqValidity(char[][] oldBoard, Sequence seq){
		boolean isValid = false;
		while(isValid == false){
			//returns true if there's an old letter at the row above the sequence;
			if(seq.startingPoint.row != 0){
				for(int i=seq.startingPoint.col; i<=seq.endingPoint.col; i++){
					if(oldBoard[seq.startingPoint.row-1][i] != '0'){
						isValid = true;
					}
				}
			}
			//returns true if there's an old letter at the row below the sequence;
			if(seq.startingPoint.row != Board.bHeight-1){
				for(int i=seq.startingPoint.col; i<=seq.endingPoint.col; i++){
					if(oldBoard[seq.startingPoint.row+1][i] != '0'){
						isValid = true;
					}
				}
			}
			//returns true if there's an old letter left to the sequence;
			if(seq.startingPoint.col != 0){
					if(oldBoard[seq.startingPoint.row][seq.startingPoint.col-1] != '0'){
						isValid = true;
					}
			}
			//returns true if there's an old letter right to the sequence;
			if(seq.endingPoint.col != Board.bWidth-1){
					if(oldBoard[seq.startingPoint.row][seq.endingPoint.col+1] != '0'){
						isValid = true;
					}
			}
			break;
		}
		return isValid;
	}

	//returns true if a every sequence is connected by at least one old letter.
	private static boolean SeqIsLegal(char[][] oldBoard, Vector<Sequence> sequences){
		boolean isValid = true;
		if(sequences.size()== 0)
			isValid = true;
		else if(sequences.size()== 1)
			isValid = oneSeqValidity(oldBoard, sequences.get(0));
		else{
			//returns false if there is a gap between two sequences.
			int row = sequences.get(0).startingPoint.row;
			for(int i=0; i<sequences.size()-1; i++){
				for(int col = sequences.get(i).endingPoint.col+1; 
						col < sequences.get(i+1).startingPoint.col;
						col++){
					if(oldBoard[row][col] == '0'){
						isValid = false;
						break;
					}
				}
			}	
		}
		return isValid;
	}
}


