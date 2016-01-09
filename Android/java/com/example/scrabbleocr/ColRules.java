package com.example.test1;

import java.util.Vector;

public class ColRules {
	//checks if all the new letters are at the same column and calls validSequenceMove if true.
	public static boolean validMove(char[][] oldBoard,Vector<Point> changes){
		if(changes.size() == 0)
			return true;
		for(int i=0, col = changes.firstElement().col; i<changes.size(); i++){
			if(changes.get(i).col != col)
				return false;
		}
		return validColSequence(oldBoard, changes);
	}
		
	//checks if there's a column sequence of old chars in gaps of new chars.
	private static boolean validColSequence(char[][] oldBoard, Vector<Point> changes){
		if(changes.size()==0)
			return true;
		Vector<Sequence> sequences = new Vector<Sequence>(0,1);
		createColSequences(sequences, changes);
		
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

	//creates column sequences of the new letters according to starting point and ending point.
	private static void createColSequences(Vector<Sequence> sequences, Vector<Point> changes){
		int seqStart = changes.get(0).row; //the first sequence start at the first element's רם'.
		int col = changes.get(0).col;
		
		int i = 0;
		while(i<changes.size()-1){
			if(changes.get(i).row+1 != changes.get(i+1).row){
				sequences.addElement(new Sequence(new Point(seqStart,col), new Point(changes.get(i).row, col)));
				seqStart = changes.get(i+1).row;
			}
			i++;
		}
		sequences.addElement(new Sequence(new Point(seqStart, col), new Point(changes.get(i).row, col)));
	}

	//returns true if a given sequence is connected to at least one old letter.
	private static boolean oneSeqValidity(char[][] oldBoard, Sequence seq){
		boolean isValid = false;
		while(isValid == false){
			//returns true if there's an old letter at the column left to the sequence;
			if(seq.startingPoint.col != 0){
				for(int i=seq.startingPoint.row; i <= seq.endingPoint.row; i++){
					if(oldBoard[i][seq.startingPoint.col-1] != '0'){
						isValid = true;
					}
				}
			}
			//returns true if there's an old letter at the column right to the sequence;
			if(seq.startingPoint.col != Board.bWidth-1){
				for(int i=seq.startingPoint.row; i<=seq.endingPoint.row; i++){
					if(oldBoard[i][seq.startingPoint.col+1] != '0'){
						isValid = true;
					}
				}
			}
			//returns true if there's an old letter above the sequence;
			if(seq.startingPoint.row != 0){
					if(oldBoard[seq.startingPoint.row-1][seq.startingPoint.col] != '0'){
						isValid = true;
					}
			}
			//returns true if there's an old letter right to the sequence;
			if(seq.endingPoint.row != Board.bHeight-1){
					if(oldBoard[seq.startingPoint.row+1][seq.endingPoint.col] != '0'){
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
			int col = sequences.get(0).startingPoint.col;
			for(int i=0; i<sequences.size()-1; i++){
				for(int row = sequences.get(i).endingPoint.row+1; 
						row < sequences.get(i+1).startingPoint.row;
						row++){
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
