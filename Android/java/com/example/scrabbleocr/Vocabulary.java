package com.example.test1;

import android.app.Activity;
import android.content.Context;
import android.content.res.AssetManager;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Collections;
import java.util.List;
import java.util.Vector;


public class Vocabulary extends Activity{
    int NUMBER_OF_WORDS = 113809;
    private Vector<String> words = new Vector<String>(113809,1);

	public Vocabulary(Context myContext){
        readWordsFile(myContext);
	}

	public boolean isWordLegal(String wordToFind){
		int index = Collections.binarySearch(words, wordToFind);
		if(index < 0)
			return false;
		else
			return true;
	}
	
	public void areWordsLegal(Vector<String> newWords, Vector<Boolean> wordsLegality){
		for(int i=0; i<newWords.size(); i++)
			wordsLegality.add(isWordLegal(newWords.get(i)));
	}

    // Reads words from text file
    public void readWordsFile(Context myContext){
        try {
            AssetManager assManager = myContext.getAssets();
            InputStream is = assManager.open("wordslist.txt");
            BufferedReader reader = new BufferedReader(new InputStreamReader(is, "UTF-8"));
            String line;
            while ((line=reader.readLine()) != null) {
                words.add(line);
            }
            reader.close();
        } catch (IOException e1) {
            e1.printStackTrace();
        }
    }

}
