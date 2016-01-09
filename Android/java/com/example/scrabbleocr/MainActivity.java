package com.example.test1;

import android.app.Activity;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.PixelFormat;
import android.graphics.SurfaceTexture;
import android.hardware.Camera;
import android.os.Bundle;
import android.os.Environment;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.Surface;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.List;
import java.util.Vector;

public class MainActivity extends AppCompatActivity {
    // Tesseract data (should not be here at the end..)
    public static final String OCR_DATA_PATH = Environment.getExternalStorageDirectory().toString() + "/Test1/";
    public static final String lang = "eng";

    private static final String TAG = "myApp";
    private List<String> words_list = new Vector<String>();   // words list from text file

    private Camera mCamera;
    private SurfaceTexture surfaceTexture = new SurfaceTexture(10);

    private Bitmap board_bitmap;
    private File mediaStorageDir;
    private File pictureFile;

    // Declare View-elements
    private TextView text_info;
    private Button button_run;
    private Button button_take_picture;
    private ImageView boardImageCaptured;
    private TextView board_text;

    private BoardDetector mBoardDetector;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        //requestWindowFeature(Window.FEATURE_NO_TITLE); // Remove title bar
        //getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);  // Enter full screen
        setContentView(R.layout.activity_main);
        Log.v(TAG, "onCreate");

        // Create the storage directory if it does not exist
        mediaStorageDir = new File(getExternalFilesDir(Environment.DIRECTORY_PICTURES), "MyCameraApp");
        //mediaStorageDir = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES), "MyCameraApp");
        if (! mediaStorageDir.exists()){
            Log.i(TAG, "Creating mediaStorageDir: " + mediaStorageDir.toString());
            if (! mediaStorageDir.mkdirs()){
                Log.d("MyCameraApp", "failed to create directory");
            }
        }
        pictureFile = new File(mediaStorageDir.getPath() + File.separator + "IMG_BOARD.jpg");
        Log.i(TAG, "pictureFile: " + pictureFile.toString());

        // Extract Tesseract OCR language data file
        Log.i(TAG, "Extracting tesseract language data to: " + OCR_DATA_PATH);
        extractOCRlangData();

        // Find view-elements
        text_info = (TextView) findViewById(R.id.text_info);
        button_run = (Button) findViewById(R.id.button_end_turn);
        button_take_picture = (Button) findViewById(R.id.button_take_picture);
        boardImageCaptured = (ImageView) findViewById(R.id.image_board);
        board_text = (TextView) findViewById(R.id.text_board);

        // Add a listener to the Capture button
        button_take_picture.setOnClickListener(
                new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        // get an image from the camera
                        Log.i(TAG, "Taking picture...");
                        mCamera.startPreview();  //start updating the preview surface. Preview must be started before you can take a picture.
                        mCamera.takePicture(null, null, mPicture);
                    }
                }
        );

        // Add a listener to the Run button
        button_run.setOnClickListener(
                new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        Log.i(TAG, "Making Turn...");
                        mBoardDetector.setImage(pictureFile);
                        mBoardDetector.findBoard();
                        mBoardDetector.buildGrid();
                        mBoardDetector.checkTilesOccupied();
                        mBoardDetector.runOCR();
                        mBoardDetector.drawGrid();
                        Bitmap img_result = mBoardDetector.getBoardWrappedImage();
                        boardImageCaptured.setImageBitmap(scaleDown(img_result, 512, false));
                        updateBoardText(mBoardDetector.getBoardLetters());
                        makeTurn();
                    }
                }
        );
    }

    @Override
    protected void onPause() {
        super.onPause();
        releaseCamera();              // release the camera immediately on pause event
    }

    @Override
    protected void onResume() {
        super.onResume();
        Log.i(TAG, "onResume:");
        mCamera = getCameraInstance(); // Obtain an instance of Camera from (open camera)
        try {
            mCamera.setPreviewTexture(surfaceTexture);   // Connect camera to the surfaceTexture. Without a surface, the camera will be unable to start the preview.
            Log.i(TAG, "onResume: set PreviewTexture with new mCamera");
        } catch (IOException e) {
            e.printStackTrace();
        }
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
    }

    /** A safe way to get an instance of the Camera object. */
    private Camera getCameraInstance(){
        Camera c = null;
        try {
            c = Camera.open(); // attempt to get a Camera instance
            Camera.Parameters cameraParameters = c.getParameters();
            Camera.Size cameraWantedSize = getClosestPictureSize(cameraParameters, 6e6);
            cameraParameters.setPictureSize(cameraWantedSize.width, cameraWantedSize.height);
            cameraParameters.setJpegQuality(90);
            cameraParameters.setPictureFormat(PixelFormat.JPEG);
            //cameraParameters.setPictureFormat(PixelFormat.RGB_888);
            c.setParameters(cameraParameters);
            Log.i(TAG, "Started Camera. Size: (" + cameraWantedSize.width + "," + cameraWantedSize.height + ").");
            //setCameraDisplayOrientation(this, 0, c);
        }
        catch (Exception e){
            // Camera is not available (in use or does not exist)
        }
        return c; // returns null if camera is unavailable
    }

    public void makeTurn() {
        text_info.setText("Making Turn");
        Log.v(TAG, "Calling Game.java");
        Game.main(getApplicationContext());
        // Update board text
        //String updated_str = "New Board State";
        //((TextView) findViewById(R.id.text_board)).setText(updated_str);
    }

    private void updateBoardText(List<Character> letters){
        String text = "";
        int n = 0;
        for (Character c : letters){
            text += c;
            n++;
            if (n % 8 == 0) text += String.format("%n"); // end of line each row
        }
        board_text.setText(text);
    }

    // =============================
    // OpenCV library initialization
    // =============================
    // OpenCV initialization Callback:
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {

        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    /* Now enable camera view to start receiving frames */
                    //mOpenCvCameraView.enableView();
                    mBoardDetector = new BoardDetector();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    // ===================
    //  Extract Tesseract
    // ===================
    //  Extract Tesseract language .traindata from assets into OCR_DATA_PATH
    public void extractOCRlangData(){
        String[] paths = new String[] { OCR_DATA_PATH, OCR_DATA_PATH + "tessdata/" };

        for (String path : paths) {
            File dir = new File(path);
            if (!dir.exists()) {
                if (!dir.mkdirs()) {
                    Log.v(TAG, "ERROR: Creation of directory " + path + " on sdcard failed");
                    return;
                } else {
                    Log.v(TAG, "Created directory " + path + " on sdcard");
                }
            }

        }

        // lang.traineddata file with the app (in assets folder)
        // You can get them at:
        // http://code.google.com/p/tesseract-ocr/downloads/list
        // This area needs work and optimization
        if (!(new File(OCR_DATA_PATH + "tessdata/" + lang + ".traineddata")).exists()) {
            try {
                AssetManager assetManager = getAssets();
                InputStream in = assetManager.open("tessdata/" + lang + ".traineddata");
                //GZIPInputStream gin = new GZIPInputStream(in);
                OutputStream out = new FileOutputStream(OCR_DATA_PATH + "tessdata/" + lang + ".traineddata");

                // Transfer bytes from in to out
                byte[] buf = new byte[1024];
                int len;
                //while ((lenf = gin.read(buff)) > 0) {
                while ((len = in.read(buf)) > 0) {
                    out.write(buf, 0, len);
                }
                in.close();
                //gin.close();
                out.close();

                Log.v(TAG, "Copied " + lang + " traineddata");
            } catch (IOException e) {
                Log.e(TAG, "Was unable to copy " + lang + " traineddata " + e.toString());
            }
        }
    }

    // ==============================
    //   Camera class and functions
    // ==============================
        private Camera.PictureCallback mPicture = new Camera.PictureCallback() {
        @Override
        public void onPictureTaken(byte[] data, Camera camera) {
            try {
                Log.i(TAG, "PictureCallback. Saving to: " + pictureFile.toString());
                FileOutputStream fos = new FileOutputStream(pictureFile, false);
                fos.write(data);
                fos.close();
            } catch (FileNotFoundException e) {
                Log.d(TAG, "File not found: " + e.getMessage());
            } catch (IOException e) {
                Log.d(TAG, "Error accessing file: " + e.getMessage());
            }
            Log.i(TAG, "Try to load image to ImageView..");
            if(pictureFile.exists()){
                Log.i(TAG, "Loading image to ImageView..");
                Bitmap myBitmap = BitmapFactory.decodeFile(pictureFile.getAbsolutePath());
                boardImageCaptured.setImageBitmap(scaleDown(myBitmap, 512, false));
            }
            mCamera.stopPreview();
            Log.i(TAG, "Exit PictureCallback");
        }
    };

    private void releaseCamera(){
        if (mCamera != null){
            mCamera.stopPreview();
            mCamera.release();        // release the camera for other applications
            mCamera = null;
        }
    }

    // Resize Bitmap to the maximum image size
    private Bitmap scaleDown(Bitmap realImage, float maxImageSize, boolean filter) {
        float ratio = Math.min(
                (float) maxImageSize / realImage.getWidth(),
                (float) maxImageSize / realImage.getHeight());
        int width = Math.round((float) ratio * realImage.getWidth());
        int height = Math.round((float) ratio * realImage.getHeight());

        Bitmap newBitmap = Bitmap.createScaledBitmap(realImage, width, height, filter);
        return newBitmap;
    }

        // Finds the closet available camera size to the wanted area size. (area=width*height)
    private Camera.Size getClosestPictureSize(Camera.Parameters parameters, double wantedArea) {
        Camera.Size resultSize = null;
        double resultSizeDelta = 1e10; // The difference between wanted and result size
        for (Camera.Size size : parameters.getSupportedPictureSizes()) {
            if (resultSize == null) {  // First loop
                resultSize = size;
            }
            else {
                int curArea = size.width * size.height;
                if (Math.abs(curArea - wantedArea) < resultSizeDelta) {
                    resultSize = size;
                    resultSizeDelta = Math.abs(curArea - wantedArea);
                }
            }
        }
        return(resultSize);
    }

    private Camera.Size getLargestPictureSize(Camera.Parameters parameters) {
        Camera.Size maxSize=null;
        for (Camera.Size size : parameters.getSupportedPictureSizes()) {
            if (maxSize == null) {
                maxSize=size;
            }
            else {
                int resultArea = maxSize.width * maxSize.height;
                int newArea = size.width * size.height;

                if (newArea > resultArea) {
                    maxSize = size;
                }
            }
        }
        return(maxSize);
    }

    public static void setCameraDisplayOrientation(Activity activity,
                                                   int cameraId, android.hardware.Camera camera) {
        android.hardware.Camera.CameraInfo info =
                new android.hardware.Camera.CameraInfo();
        android.hardware.Camera.getCameraInfo(cameraId, info);
        int rotation = activity.getWindowManager().getDefaultDisplay().getRotation();
        int degrees = 0;
        switch (rotation) {
            case Surface.ROTATION_0: degrees = 0; break;
            case Surface.ROTATION_90: degrees = 90; break;
            case Surface.ROTATION_180: degrees = 180; break;
            case Surface.ROTATION_270: degrees = 270; break;
        }

        int result;
        if (info.facing == Camera.CameraInfo.CAMERA_FACING_FRONT) {
            result = (info.orientation + degrees) % 360;
            result = (360 - result) % 360;  // compensate the mirror
        } else {  // back-facing
            result = (info.orientation - degrees + 360) % 360;
        }
        camera.setDisplayOrientation(result);
    }
}
