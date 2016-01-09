package com.example.test1;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Environment;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;

import java.io.File;
import java.util.List;
import java.util.Vector;

import com.googlecode.tesseract.android.TessBaseAPI;

public class BoardDetector {
    // Params
    private static final String TAG = "BoardDetectorLog";
    private static final int BOARD_ROWS = 8;
    private static final int BOARD_COLS = 8;
    private static final double MAX_DIST_FROM_CENTER = 400; // Maximum allowed distance (pixels) of detected board centroid from center
    //private static final double MIN_RECT_AREA = 15000; // Minimum area of board contour (in pixels)
    private static final int PyramidLevels = 2;
    private static final int IMG_WARPED_SIZE_X = 1024; // result quad x in pixels
    private static final int IMG_WARPED_SIZE_Y = 1024; // result quad y in pixels
    private static final int TEMPLATE_SIZE = 32;  // Template for finding corners of board, in pixels.
    private static final double OCCUPIED_STD_TH = 15; // STD threshold value to decide it tile is empty or has letter in it

    // Tesseract data:
    public static final String OCR_DATA_PATH = Environment.getExternalStorageDirectory().toString() + "/Test1/";
    public static final String lang = "eng";

    // Cache
    private Bitmap mImageBitmap;        // Android image Bitmap
    private Mat mImageRgbMat = new Mat();  // OpenCV RGB image Mat
    private Mat mImageHsvMat = new Mat();  // OpenCV HSV image Mat
    private Mat mImageBwMat = new Mat();  // OpenCV Black-White image Mat
    private Mat mPyrDownMat = new Mat();  // OpenCV Mat of down-sampled RGB image
    private Mat mRGBImageWarped = new Mat();  // OpenCV Image after projective transformation
    private Bitmap mRGBWarpedBitmap;  // Android Bitmap Image after projective transformation
    private Vector<Tile> tilesGrid = new Vector<>(BOARD_ROWS*BOARD_COLS);


    private class Tile{
        public Tile(int row, int column, Point p1, Point p2){
            this.row = row;
            this.column = column;
            this.p1 = p1;
            this.p2 = p2;
            this.occupied = false;
            this.letter = '0';
        }
        public Tile(int row, int column, Point p1, Point p2, boolean occupied, char letter){
            this.row = row;
            this.column = column;
            this.p1 = p1;
            this.p2 = p2;
            this.occupied = occupied;
            this.letter = letter;
        }
        public Rect getRect(){
            return new Rect(this.p1, this.p2);
        }
        public boolean isOccupied(){
            return this.occupied;
        }
        public int row, column;  // board row, column
        public Point p1, p2; // top-left, bottom-right
        public boolean occupied = false;
        public char letter = '0';
    }

    public void setImage(File boardImageFile){
        Log.v(TAG, "Loading image from: " + boardImageFile.getAbsolutePath());
        if(boardImageFile.exists()) {
            mImageBitmap = BitmapFactory.decodeFile(boardImageFile.getAbsolutePath());
        }
        else{
            Log.i(TAG, "Can't find image filename");
            return;
        }
        // Convert Bitmap to Mat
        Utils.bitmapToMat(mImageBitmap, mImageRgbMat);
        Imgproc.cvtColor(mImageRgbMat, mImageRgbMat, Imgproc.COLOR_RGBA2RGB);
    }

    public Bitmap getBoardWrappedImage(){
        //boardImageCaptured.setImageBitmap(scaleDown(img_result, 512, false));
        if (!mRGBImageWarped.empty()){
            return convertMatToBitmap(mRGBImageWarped);
        } else {
            Log.i(TAG, "Image warped does not exist.");
            return convertMatToBitmap(mImageBwMat);
        }
    }

    public List<Character> getBoardLetters(){
        List<Character> result = new Vector<>(BOARD_ROWS*BOARD_COLS);
        for (Tile tile : tilesGrid){
            result.add(tile.letter);
        }
        return result;
    }

    private Bitmap convertMatToBitmap(Mat imageMat){
        if (imageMat.channels()==1) Imgproc.cvtColor(imageMat, imageMat, Imgproc.COLOR_GRAY2RGBA);
        else if (imageMat.channels()==3) Imgproc.cvtColor(imageMat, imageMat, Imgproc.COLOR_RGB2RGBA);
        Bitmap img_result = Bitmap.createBitmap(imageMat.cols(), imageMat.rows(), Bitmap.Config.ARGB_8888);  //RGB_565
        Utils.matToBitmap(imageMat, img_result, true);
        return img_result;
    }

    public void drawGrid(){
        Scalar cyan = new Scalar(0,255,255);
        Scalar red = new Scalar(255,0,0);
        Scalar color;
        for (Tile tile : tilesGrid){
            if (tile.isOccupied()) color = red;
            else color = cyan;
            Imgproc.rectangle(mRGBImageWarped, tile.p1, tile.p2, color, 5);
        }
    }

    // Gets an image of the board, finds the board, do a perspective transformation and find the letters.
    public void findBoard(){
        // Blur
        //Size blur_kernel = new Size(5, 5);
        //Imgproc.GaussianBlur(mImageRgbMat, mImageRgbMat, blur_kernel, 0, 0);

        // Down-sample original image by 4 using image pyramids
        Imgproc.pyrDown(mImageRgbMat, mPyrDownMat);
        for (int i=1; i < PyramidLevels; i++){
            Imgproc.pyrDown(mPyrDownMat, mPyrDownMat);
        }

        // Apply Threshold
        Imgproc.cvtColor(mPyrDownMat, mImageHsvMat, Imgproc.COLOR_RGB2HSV);
        Scalar low1 = new Scalar(0, 100, 50);
        Scalar high1 = new Scalar(10, 230, 235);
        Scalar low2 =  new Scalar(170, 100, 50);
        Scalar high2 = new Scalar(179, 230, 235);
        Mat bw1 = new Mat(); // ((int) image.getHeight(), (int) image.getWidth(), CvType.CV_8UC1);
        Mat bw2 = new Mat(); // ((int) image.getHeight(), (int) image.getWidth(), CvType.CV_8UC1);
        Core.inRange(mImageHsvMat, low1, high1, bw1);
        Core.inRange(mImageHsvMat, low2, high2, bw2);
        Core.bitwise_or(bw1, bw2, mImageBwMat);
        Size morph_size = new Size(7,7);
        Mat element = Imgproc.getStructuringElement(Imgproc.CV_SHAPE_RECT, morph_size);
        Imgproc.morphologyEx(mImageBwMat, mImageBwMat, Imgproc.MORPH_OPEN, element);
        Imgproc.morphologyEx(mImageBwMat, mImageBwMat, Imgproc.MORPH_CLOSE, element);

        // Find Contours
        Log.v(TAG, "Finding contour of board...");
        List<MatOfPoint> contours = new Vector<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(mImageBwMat.clone(), contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        // Find large rectangle contour
        MatOfPoint selectedContour = new MatOfPoint();
        double imageSize = mPyrDownMat.width()*mPyrDownMat.height();
        for (MatOfPoint contour : contours) {
            double curArea = Math.abs(Imgproc.contourArea(contour));
            if (curArea > imageSize/16) selectedContour = contour; // contour area needs to be at least 1/8 of the image
            else Log.d(TAG, "Contour Area=" + curArea + " , Required area=" + imageSize/16);
        }
        if (selectedContour.empty()){
            Log.i(TAG, "No Contour was found, or contour was too small");
            return;
        }
        Core.multiply(selectedContour, new Scalar(Math.pow(2,PyramidLevels), Math.pow(2,PyramidLevels)), selectedContour); // resize contour back to original image size

        // Find minimum rotated bounding rect
        MatOfPoint2f contours_float = new MatOfPoint2f( selectedContour.toArray() );
        RotatedRect minRect = Imgproc.minAreaRect(contours_float);

        // make sure bounding rect is around image center
        double centerImgPoint_x = mImageRgbMat.width()/2;
        double centerImgPoint_y = mImageRgbMat.height()/2;
        double dist_x_pow = Math.pow(minRect.center.x - centerImgPoint_x,2);
        double dist_y_pow = Math.pow(minRect.center.y - centerImgPoint_y, 2);
        double distFromCenter = Math.sqrt(dist_x_pow + dist_y_pow);
        if (distFromCenter > MAX_DIST_FROM_CENTER){
            Log.i(TAG, "Bounding Rect centroid is too far from center. dist=" + distFromCenter + " > MAX_DIST_FROM_CENTER=" + MAX_DIST_FROM_CENTER);
        }

        // Find 4 corners
        List<Point> corners = findCorners(selectedContour.toList());

        // Apply Projective transformation
        Log.i(TAG, "Applying Projective transformation");
        mRGBImageWarped = perspectiveTransformation(mImageRgbMat, corners);
        mRGBWarpedBitmap = convertMatToBitmap(mRGBImageWarped.clone());
        mRGBWarpedBitmap = mRGBWarpedBitmap.copy(Bitmap.Config.ARGB_8888, true);

    }

    public void buildGrid(){
        List<Point> corners = findBoardCorners();
        double vr_x = (corners.get(1).x - corners.get(0).x) / BOARD_COLS; // one unit of vector right
        double vr_y = (corners.get(1).y - corners.get(0).y) / BOARD_ROWS; // one unit of vector right
        double vd_x = (corners.get(3).x - corners.get(0).x) / BOARD_COLS; // one unit of vector down
        double vd_y = (corners.get(3).y - corners.get(0).y) / BOARD_ROWS; // one unit of vector down

        tilesGrid.clear();
        for (int row=0; row < BOARD_ROWS; row++){
            for (int col=0; col < BOARD_COLS; col++){
                Point p1 = new Point((int) Math.round(corners.get(0).x + col * vr_x + row * vd_x),
                                     (int) Math.round(corners.get(0).y + col * vr_y + row * vd_y));
                Point p2 = new Point((int) Math.round(corners.get(0).x + (col + 1) * vr_x + (row + 1) * vd_x),
                                     (int) Math.round(corners.get(0).y + (col + 1) * vr_y + (row + 1) * vd_y));
                tilesGrid.add(new Tile(row, col, p1, p2, false, '0'));
                Log.d(TAG, "Tile added: P1=" + p1.x + "," + p1.y + " P2=" + p2.x + "," + p2.y);
            }
        }

    }

    public void checkTilesOccupied(){
        Core.MinMaxLocResult mmr;
        Mat cur_tile = new Mat();
        for (Tile tile : tilesGrid){
            // TODO - check if tile is occupied
            //MatOfDouble mean = new MatOfDouble();
            //MatOfDouble std = new MatOfDouble();
            //Core.meanStdDev(mRGBImageWarped.submat(tile.getRect()), mean, std);
            //Log.i(TAG, "STD " + tile.row + "," + tile.column + ":" + std.get(0, 0)[0]);
            //if (std.get(0,0)[0] > OCCUPIED_STD_TH){
            //    tile.occupied = true;
            //    Log.i(TAG, "Tile (" + tile.row + "," + tile.column + ") is occupied.");
            //}
            Imgproc.cvtColor(mRGBImageWarped.submat(tile.getRect()), cur_tile, Imgproc.COLOR_RGB2GRAY);
            mmr = Core.minMaxLoc(cur_tile);
            Log.i(TAG, "PTP: " + (mmr.maxVal - mmr.minVal));
            if (mmr.maxVal - mmr.minVal > 120){
                tile.occupied = true;
                Log.i(TAG, "Tile (" + tile.row + "," + tile.column + ") is occupied.");
            }
        }
    }

    public void runOCR(){
        Log.v(TAG, "Initialize Tesseract OCR");
        TessBaseAPI baseApi = new TessBaseAPI();
        //baseApi.setDebug(true);
        baseApi.init(OCR_DATA_PATH, lang, TessBaseAPI.OEM_DEFAULT); //OEM_TESSERACT_ONLY
        baseApi.setPageSegMode(TessBaseAPI.PageSegMode.PSM_SINGLE_CHAR);
        baseApi.setVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ");
        int x,y,width,height;
        Bitmap tile_img;
        for (Tile tile : tilesGrid){
            // TODO - check only if tile is occupied
            if (tile.isOccupied()) {
                x = (int) (tile.p1.x + 10);
                y = (int) (tile.p1.y + 10);
                width = (int) (tile.p2.x - tile.p1.x - 20);
                height = (int) (tile.p2.y - tile.p1.y - 20);
                tile_img = Bitmap.createBitmap(mRGBWarpedBitmap, x, y, width, height);
                tile_img = Bitmap.createScaledBitmap(tile_img, width / 4, height / 4, true);  // Tesseract don't handle good extra large text
                baseApi.setImage(tile_img);
                String recognizedLetter = baseApi.getUTF8Text();
                int conf = baseApi.meanConfidence();
                Log.v(TAG, "(" + tile.row + "," + tile.column + ") OCR Letter: " + recognizedLetter + "(Confidence=" + conf + "%)");
                if (conf > 75) {
                    tile.letter = recognizedLetter.charAt(0);
                }
            }
        }
        baseApi.end();
    }

    // Finds board accurate corners using correlation
    private List<Point> findBoardCorners(){
        Mat template = createTemplate(0); // top-left corner template
        Mat corr_result = new Mat();
        List<Point> corners = new Vector<>(4);
        Core.MinMaxLocResult mmr;
        int roi_img_size_x = IMG_WARPED_SIZE_X / 8;
        int roi_img_size_y = IMG_WARPED_SIZE_Y / 8;

        // find top-left corner
        Imgproc.matchTemplate(mRGBImageWarped.submat(0,roi_img_size_x,0,roi_img_size_y),
                template, corr_result, Imgproc.TM_CCOEFF_NORMED);
        mmr = Core.minMaxLoc(corr_result);
        Point topLeft = new Point((int) mmr.maxLoc.x + TEMPLATE_SIZE /2,
                                  (int) mmr.maxLoc.y + TEMPLATE_SIZE /2);

        // find bottom-right
        Core.flip(template, template, -1);
        int roi_col = mRGBImageWarped.width()-roi_img_size_x;
        int roi_row = mRGBImageWarped.height()-roi_img_size_y;
        Imgproc.matchTemplate(mRGBImageWarped.submat(roi_row, mRGBImageWarped.height(), roi_col,  mRGBImageWarped.width()),
                template, corr_result, Imgproc.TM_CCOEFF_NORMED);
        mmr = Core.minMaxLoc(corr_result);
        Point bottomRight = new Point(roi_col + (int) mmr.maxLoc.x + TEMPLATE_SIZE /2,
                                      roi_row + (int) mmr.maxLoc.y + TEMPLATE_SIZE /2);

        // find two other corners by calculation
        double xc = (topLeft.x + bottomRight.x)/2;
        double yc = (topLeft.y + bottomRight.y)/2;    // Center point
        double xd = (topLeft.x - bottomRight.x)/2;
        double yd = (topLeft.y - bottomRight.y)/2;    // Half-diagonal
        Point topRight = new Point(xc - yd, yc + xd);
        Point bottomLeft = new Point(xc + yd, yc - xd);

        corners.add(topLeft);
        corners.add(topRight);
        corners.add(bottomRight);
        corners.add(bottomLeft);
        return corners;
    }

    // Create template for the top-left corner of board, to use with matchTemplate to find the corners.
    private Mat createTemplate(int type){
        // type: 0-top left, 1-top right, 2-bottom right, 3-bottom left
        byte[] red = new byte[]{127, 0, 0};
        byte[] lightblue = new byte[]{62, 88, 111};
        Mat rgb = new Mat(TEMPLATE_SIZE, TEMPLATE_SIZE, CvType.CV_8UC3, new Scalar(0,0,0));
        for (int row=0 ; row < TEMPLATE_SIZE; row++){
            for (int col=0 ; col < TEMPLATE_SIZE; col++){
                if ( (col<(TEMPLATE_SIZE/2-3)) || (col>=(TEMPLATE_SIZE/2-3) && row<=(TEMPLATE_SIZE/2-3)) )
                    rgb.put(row, col,red); // red
                if ( col>=(TEMPLATE_SIZE/2+3) && row>=(TEMPLATE_SIZE/2+3) )
                    rgb.put(row, col, lightblue);
            }
        }
        if (type==1)  // Top-right
            Core.flip(rgb, rgb, 1);
        if (type==2)  // Bottom-right
            Core.flip(rgb, rgb, -1);
        if (type==3)  // Bottom-left
            Core.flip(rgb, rgb, 0);
        return rgb;
    }




    // Sorts the 4 corners of the quadrilateral, assuming it is not distorted and closed to the image edges
    // The first one will be the top-left and then clockwise.
    // The input is the current vector of 4 corners (unsorted) and
    private List<Point> findCorners(List<Point> points){
        List<Point> corners = new Vector<>(4);
        Point tl = points.get(0);
        Point tr = points.get(0);
        Point br = points.get(0);
        Point bl = points.get(0);
        double min_tl, min_br, min_tr, min_bl;
        min_tl = points.get(0).x + points.get(0).y;
        min_br = - points.get(0).x - points.get(0).y;
        min_tr = - points.get(0).x + points.get(0).y;
        min_bl = points.get(0).x - points.get(0).y;
        for (Point p : points){
            if (p.x + p.y < min_tl){
                min_tl = p.x + p.y;
                tl = p.clone();
            }
            if (-p.x - p.y < min_br){
                min_br = -p.x - p.y;
                br = p.clone();
            }
            if (-p.x + p.y < min_tr){
                min_tr = -p.x + p.y;
                tr = p.clone();
            }
            if (p.x - p.y < min_bl){
                min_bl = p.x - p.y;
                bl = p.clone();
            }
        }
        corners.add(tl);
        corners.add(tr);
        corners.add(br);
        corners.add(bl);
        return corners;
    }

    // Apply perspective transformation on the found quadrilateral.
    // Gets an image and 4 corners and returns the quadrilateral
    // ROI as square.
    private Mat perspectiveTransformation(Mat img, List<Point> corners) {
        // Define the destination image
        Mat quad = Mat.zeros(IMG_WARPED_SIZE_X, IMG_WARPED_SIZE_Y, CvType.CV_8UC3);

        // Corners of the destination image
        List<org.opencv.core.Point> quad_pts = new Vector<>(4);
        quad_pts.add(new Point(0, 0));
        quad_pts.add(new Point((float) quad.cols(), 0));
        quad_pts.add(new Point((float) quad.cols(), (float) quad.rows()));
        quad_pts.add(new Point(0, (float) quad.rows()));

        // Get transformation matrix
        Mat cornersMat = Converters.vector_Point_to_Mat(corners, CvType.CV_32F);
        Mat quadMat = Converters.vector_Point_to_Mat(quad_pts, CvType.CV_32F);
        Mat transformMatrix = Imgproc.getPerspectiveTransform(cornersMat, quadMat);

        // Apply perspective transformation
        Imgproc.warpPerspective(img, quad, transformMatrix, quad.size());
        return quad;
    }


    // Sorts the 4 corners of the quadrilateral.
    // The first one will be the top-left and then clockwise.
    // The input is the current vector of 4 corners (unsorted) and
    // the center of mass of the quadrilateral.
    private List<Point> sortCorners(List<Point> corners, Point center) {
        List<Point> sortedCorners = new Vector<>(4);
        List<Point> top = new Vector<>();
        List<Point> bot = new Vector<>();
        for (int i = 0; i < corners.size(); i++)
        {
            if (corners.get(i).y < center.y)
                top.add(corners.get(i));
            else
                bot.add(corners.get(i));
        }
        Point tl = top.get(0).x > top.get(1).x ? top.get(1) : top.get(0);
        Point tr = top.get(0).x > top.get(1).x ? top.get(0) : top.get(1);
        Point bl = bot.get(0).x > bot.get(1).x ? bot.get(1) : bot.get(0);
        Point br = bot.get(0).x > bot.get(1).x ? bot.get(0) : bot.get(1);
        sortedCorners.add(tl);
        sortedCorners.add(tr);
        sortedCorners.add(br);
        sortedCorners.add(bl);
        return sortedCorners;
    }

}
