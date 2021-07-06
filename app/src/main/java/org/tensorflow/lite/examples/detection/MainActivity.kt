package org.tensorflow.lite.examples.detection

import android.app.Activity
import android.content.Intent
import android.graphics.*
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.os.Handler
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import org.tensorflow.lite.examples.detection.customview.OverlayView
import org.tensorflow.lite.examples.detection.customview.OverlayView.DrawCallback
import org.tensorflow.lite.examples.detection.env.ImageUtils
import org.tensorflow.lite.examples.detection.env.Logger
import org.tensorflow.lite.examples.detection.env.Utils
import org.tensorflow.lite.examples.detection.tflite.Classifier
import org.tensorflow.lite.examples.detection.tflite.Classifier.Recognition
import org.tensorflow.lite.examples.detection.tflite.YoloV4Classifier
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker
import java.io.File
import java.io.IOException
import java.io.InputStream
import java.text.SimpleDateFormat
import java.util.*

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        cameraButton = findViewById(R.id.cameraButton)
        detectButton = findViewById(R.id.detectButton)
        imageView = findViewById(R.id.imageView)
        textView = findViewById(R.id.textView)
//        cameraButton?.setOnClickListener(View.OnClickListener { v: View? -> startActivity(Intent(this@MainActivity, DetectorActivity::class.java)) })
        cameraButton?.setOnClickListener {
            dispatchTakePictureIntent()
        }
        detectButton?.setOnClickListener(View.OnClickListener { v: View? ->
            val handler = Handler()
            Thread(Runnable {
                val results = detector!!.recognizeImage(cropBitmap)
                handler.post(Runnable { handleResult(cropBitmap, results) })
            }).start()
        })
        sourceBitmap = Utils.getBitmapFromAsset(this@MainActivity, "colgate_31.jpg")
        cropBitmap = Utils.processBitmap(sourceBitmap, TF_OD_API_INPUT_SIZE)
        imageView?.setImageBitmap(cropBitmap)
        initBox()
    }

    private val sensorOrientation = 90
    private var detector: Classifier? = null
    private var frameToCropTransform: Matrix? = null
    private var cropToFrameTransform: Matrix? = null
    private var tracker: MultiBoxTracker? = null
    private var trackingOverlay: OverlayView? = null
    protected var previewWidth = 0
    protected var previewHeight = 0
    private var sourceBitmap: Bitmap? = null
    private var cropBitmap: Bitmap? = null
    private var cameraButton: Button? = null
    private var detectButton: Button? = null
    private var imageView: ImageView? = null
    private var textView: TextView? = null
    private fun initBox() {
        previewHeight = TF_OD_API_INPUT_SIZE
        previewWidth = TF_OD_API_INPUT_SIZE
        frameToCropTransform = ImageUtils.getTransformationMatrix(
                previewWidth, previewHeight,
                TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE,
                sensorOrientation, MAINTAIN_ASPECT)
        cropToFrameTransform = Matrix()
        frameToCropTransform?.invert(cropToFrameTransform)
        tracker = MultiBoxTracker(this)
        trackingOverlay = findViewById(R.id.tracking_overlay)
        trackingOverlay?.addCallback(
                DrawCallback { canvas: Canvas? -> tracker!!.draw(canvas) })
        tracker!!.setFrameConfiguration(TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE, sensorOrientation)
        try {
            detector = YoloV4Classifier.create(
                    assets,
                    TF_OD_API_MODEL_FILE,
                    TF_OD_API_LABELS_FILE,
                    TF_OD_API_IS_QUANTIZED)
        } catch (e: IOException) {
            e.printStackTrace()
            LOGGER.e(e, "Exception initializing classifier!")
            val toast = Toast.makeText(
                    applicationContext, "Classifier could not be initialized", Toast.LENGTH_SHORT)
            toast.show()
            finish()
        }
    }

    private fun handleResult(bitmap: Bitmap?, results: List<Recognition>) {
        val canvas = Canvas(bitmap!!)
        val paint = Paint()
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 2.0f
        val textPaint = Paint()
        textPaint.color = Color.WHITE
        textPaint.textSize = 20.0f
        val mappedRecognitions: List<Recognition> = LinkedList()
        var cavity = 0
        var maxfresh = 0
        var colgate = 0
        var sparkle = 0
        var total = 0
        var triple = 0
        for (result in results) {
            val location = result.location
            if (location != null && result.confidence >= MINIMUM_CONFIDENCE_TF_OD_API) {
                var detectedClass = ""
                var color = 0
                when (result.detectedClass) {
                    0 -> {
                        color = Color.GREEN
                        detectedClass = "Cavity"
                        cavity++
                    }
                    1 -> {
                        color = Color.BLACK
                        detectedClass = "Colgate"
                        colgate++
                    }
                    2 -> {
                        color = Color.CYAN
                        detectedClass = "MaxFresh"
                        maxfresh++
                    }
                    3 -> {
                        color = Color.YELLOW
                        detectedClass = "Sparkle"
                        sparkle++
                    }
                    4 -> {
                        color = Color.MAGENTA
                        detectedClass = "Total"
                        total++
                    }
                    5 -> {
                        color = Color.RED
                        detectedClass = "triple"
                        triple++
                    }
                }
                paint.color = color
                canvas.drawText(detectedClass, location.left, location.top, textPaint)
                canvas.drawRect(location, paint)
            }
        }
        //        tracker.trackResults(mappedRecognitions, new Random().nextInt());
//        trackingOverlay.postInvalidate();
        textView!!.text = "Colgate : $colgate\nMax-Fresh :$maxfresh\ntriple :$triple\ntotal :$total\nsparkle: $sparkle\ncavity: $cavity"
        imageView!!.setImageBitmap(bitmap)
    }

    companion object {
        const val MINIMUM_CONFIDENCE_TF_OD_API = 0.5f
        private val LOGGER = Logger()
        const val TF_OD_API_INPUT_SIZE = 416
        private const val TF_OD_API_IS_QUANTIZED = false
        private const val TF_OD_API_MODEL_FILE = "yolov4-416-fp32.tflite"
        private const val TF_OD_API_LABELS_FILE = "file:///android_asset/coco.txt"

        // Minimum detection confidence to track a detection.
        private const val MAINTAIN_ASPECT = false
        const val REQUEST_IMAGE_CAPTURE: Int = 1
    }

    private fun dispatchTakePictureIntent() {

        Intent(MediaStore.ACTION_IMAGE_CAPTURE).also { takePictureIntent ->
            // Ensure that there's a camera activity to handle the intent
            takePictureIntent.resolveActivity(packageManager)?.also {
                // Create the File where the photo should go
                val photoFile: File? = try {
                    createImageFile()
                } catch (e: IOException) {
                    Log.e("Image Capture", e.message.toString())
                    null
                }
                // Continue only if the File was successfully created
                photoFile?.also {
                    val photoURI: Uri = FileProvider.getUriForFile(
                            this,
                            "org.tensorflow.lite.examples.detection.fileprovider",
                            it
                    )
                    takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI)
                    startActivityForResult(takePictureIntent, Companion.REQUEST_IMAGE_CAPTURE)
                }
            }
        }
    }
    private fun createImageFile(): File {
        // Create an image file name
        val timeStamp: String = SimpleDateFormat("yyyyMMdd_HHmmss").format(Date())
        val storageDir: File? = getExternalFilesDir(Environment.DIRECTORY_PICTURES)
        return File.createTempFile(
                "JPEG_${timeStamp}_", /* prefix */
                ".jpg", /* suffix */
                storageDir /* directory */
        ).apply {
            // Save a file: path for use with ACTION_VIEW intents
            currentPhotoPath = absolutePath
        }
    }
    private lateinit var currentPhotoPath: String
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQUEST_IMAGE_CAPTURE &&
                resultCode == Activity.RESULT_OK
        ) {
            getCapturedImage()
        }
    }
    /**
     * getCapturedImage():
     *     Decodes and crops the captured image from camera.
     */
    private fun getCapturedImage() {

        var istr: InputStream
        sourceBitmap = Utils.getBitmapFromAsset(this@MainActivity, currentPhotoPath)
        sourceBitmap = BitmapFactory.decodeFile(currentPhotoPath)
        cropBitmap = Utils.processBitmap(sourceBitmap, TF_OD_API_INPUT_SIZE)
        val handler = Handler()
        Thread(Runnable {
            val results = detector!!.recognizeImage(cropBitmap)
            handler.post(Runnable { handleResult(cropBitmap, results) })
        }).start()
    }

}