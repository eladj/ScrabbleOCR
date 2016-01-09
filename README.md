Scrabble OCR using OpenCV and Tesseract-OCR
===========================================


In this repository there are 3 implementations:

- **Android-Java:** Basic implementation including image capture and game managment. The Android app game management is not finished, but the computer vision and OCR part is working.
- **C++:** Basic OCR prototype. Gets an input image and retrives the letters on the board.
- **Python:** Basic OCR prototype. Gets an input image and retrives the letters on the board.


Basic Algorithm
---------------

1. Detect the outer frame of the board according to its red color, using HSV color coordinates.
2. Find the 4 corners of the board and arrange them by order.
3. Apply perspective transformation to compensate the camera view angle.
4. Detect each tile square and decide if it is empty or have a letter on it, by looking on the STD or gradients strength. 
5. For each letter we found, we call the Tesseract optical character recongnition.

RGB and HSV Channels:

![HSV image](/images/scrabble_img2.jpg)

Board detection:

![Board detection](/images/scrabble_img1.jpg)
 

Screenshots
-----------

Android app:

![My image](/images/scrabble_img3.jpg)
![My image](/images/scrabble_img4.jpg)
