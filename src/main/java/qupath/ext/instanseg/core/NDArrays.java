package qupath.ext.instanseg.core;

import java.awt.image.BufferedImage;
import java.awt.image.DataBuffer;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URISyntaxException;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.ShortBuffer;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;
import org.apposed.appose.Appose;
import org.apposed.appose.BuildException;
import org.apposed.appose.Environment;
import org.apposed.appose.NDArray;
import org.apposed.appose.Service;
import org.apposed.appose.TaskException;
import org.bytedeco.javacpp.ShortPointer;
import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.opencv_core.Mat;
import org.opencv.core.CvType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.lib.common.ColorTools;
import qupath.lib.experimental.pixels.OutputHandler;
import qupath.lib.experimental.pixels.Parameters;
import qupath.lib.gui.scripting.QPEx;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.objects.PathCellObject;
import qupath.lib.regions.RegionRequest;
import qupath.opencv.tools.OpenCVTools;

public class NDArrays {
    private NDArrays() {
        throw new UnsupportedOperationException("Do not instantiate this class");
    }
    private static final Logger logger = LoggerFactory.getLogger(NDArrays.class);

    public static NDArray fromBufferedImage(ImageServer<BufferedImage> server, RegionRequest regionRequest) throws IOException {
        var img = server.readRegion(regionRequest);
        switch (img.getType()) {
            case BufferedImage.TYPE_INT_BGR:
            case BufferedImage.TYPE_3BYTE_BGR:
                return imageToNDArrayRGBorBGR(img, true, false);
            case BufferedImage.TYPE_4BYTE_ABGR:
            case BufferedImage.TYPE_4BYTE_ABGR_PRE:
                return imageToNDArrayRGBorBGR(img, true, true);
            case BufferedImage.TYPE_INT_ARGB:
            case BufferedImage.TYPE_INT_ARGB_PRE:
                return imageToNDArrayRGBorBGR(img, false, true);
            case BufferedImage.TYPE_USHORT_555_RGB:
            case BufferedImage.TYPE_USHORT_565_RGB:
            case BufferedImage.TYPE_INT_RGB:
                return imageToNDArrayRGBorBGR(img, false, false);
            case BufferedImage.TYPE_USHORT_GRAY:
        }

        var dtype = getDType(img.getRaster().getDataBuffer());
        NDArray ndArray = new NDArray(
                dtype,
                new NDArray.Shape(
                        NDArray.Shape.Order.C_ORDER, // todo: figure out which to use when. currently assume C for numpy-ish and F for imglib2-ish code
                        img.getHeight(),
                        img.getWidth(),
                        img.getRaster().getNumBands()
                )
        );

        // copy one buffer to another
        var raster = img.getRaster();
        var width = img.getWidth();
        var height = img.getHeight();
        var dataBuffer = raster.getDataBuffer();
        int dataType = dataBuffer.getDataType();
        int bands = raster.getNumBands();

        var buffer = ndArray.buffer();
        switch (dataType) {
            case DataBuffer.TYPE_BYTE -> {
                int[] pixel = new int[bands];
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        raster.getPixel(x, y, pixel);
                        for (int b = 0; b < bands; b++) {
                            buffer.put((byte) pixel[b]);
                        }
                    }
                }
            }
            case DataBuffer.TYPE_USHORT, DataBuffer.TYPE_SHORT -> {
                int[] pixel = new int[bands];
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        raster.getPixel(x, y, pixel);
                        for (int b = 0; b < bands; b++) {
                            buffer.putShort((short) pixel[b]);
                        }
                    }
                }
            }
            case DataBuffer.TYPE_INT -> {
                int[] pixel = new int[bands];
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        raster.getPixel(x, y, pixel);
                        for (int b = 0; b < bands; b++) {
                            buffer.putInt(pixel[b]);
                        }
                    }
                }
            }
            case DataBuffer.TYPE_FLOAT -> {
                float[] pixel = new float[bands];
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        raster.getPixel(x, y, pixel);
                        for (int b = 0; b < bands; b++) {
                            buffer.putFloat(pixel[b]);
                        }
                    }
                }
            }
            case DataBuffer.TYPE_DOUBLE -> {
                double[] pixel = new double[bands];
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        raster.getPixel(x, y, pixel);
                        for (int b = 0; b < bands; b++) {
                            buffer.putDouble(pixel[b]);
                        }
                    }
                }
            }
            default -> throw new IllegalArgumentException("Unsupported DataBuffer type: " + dataType);
        }
        return ndArray;
    }

    private static NDArray.DType getDType(DataBuffer buffer) {
        int type = buffer.getDataType();
        return switch(type) {
            case DataBuffer.TYPE_BYTE -> NDArray.DType.INT8;
            case DataBuffer.TYPE_SHORT -> NDArray.DType.INT16;
            case DataBuffer.TYPE_USHORT -> NDArray.DType.UINT16;
            case DataBuffer.TYPE_INT -> NDArray.DType.INT32;
            case DataBuffer.TYPE_FLOAT -> NDArray.DType.FLOAT32;
            case DataBuffer.TYPE_DOUBLE -> NDArray.DType.FLOAT64;
            default -> throw new RuntimeException("Unknown type: " + type);
        };
    }

    private static NDArray.DType getDType(Mat mat) {
		return switch (mat.depth()) {
			case opencv_core.CV_8U -> NDArray.DType.UINT8;
			case opencv_core.CV_8S -> NDArray.DType.INT8; // Byte is unsigned
			case opencv_core.CV_16U -> NDArray.DType.UINT16;
			case opencv_core.CV_16S -> NDArray.DType.INT16;
			case opencv_core.CV_32S -> NDArray.DType.INT32;
			case opencv_core.CV_32F -> NDArray.DType.FLOAT32;
            case opencv_core.CV_64F -> NDArray.DType.FLOAT64;
            default -> {
                logger.warn("Unknown Mat depth {}, will default to Float64 ({})", mat.depth(), NDArray.DType.FLOAT64);
                yield NDArray.DType.FLOAT64;
            }
        };
    }


    private static NDArray imageToNDArrayRGBorBGR(final BufferedImage img, final boolean doBGR, final boolean includeAlpha) {
        int width = img.getWidth();
        int height = img.getHeight();
        int[] data = img.getRGB(0, 0, width, height, null, 0, img.getWidth());

        NDArray ndArray = new NDArray(
                NDArray.DType.UINT8,
                new NDArray.Shape(
                        NDArray.Shape.Order.C_ORDER, // todo: figure out which to use when. currently assume C for numpy-ish and F for imglib2-ish code
                        img.getHeight(),
                        img.getWidth(),
                        includeAlpha ? 4 : 3
                )
        );

        var buff = ndArray.buffer();
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int val = data[y*width + x];
                int r = ColorTools.red(val);
                int g = ColorTools.green(val);
                int b = ColorTools.blue(val);
                if (doBGR) {
                    buff.put((byte) b);
                    buff.put((byte) g);
                    buff.put((byte) r);
                } else {
                    buff.put((byte) r);
                    buff.put((byte) g);
                    buff.put((byte) b);
                }
                if (includeAlpha) {
                    int a = ColorTools.alpha(val);
                    buff.put((byte) a);
                }
            }
        }
        // just in case
        buff.rewind();
        return ndArray;
    }


    public static NDArray matToNDArray(Mat mat) {

        var dtype = getDType(mat);
        NDArray ndArray = new NDArray(
                dtype,
                new NDArray.Shape(
                        NDArray.Shape.Order.C_ORDER, // todo: figure out which to use when. currently assume C for numpy-ish and F for imglib2-ish code
                        mat.rows(),
                        mat.cols(),
                        mat.channels()
                )
        );

        Buffer inBuffer = mat.createBuffer();
        ByteBuffer buffer = ndArray.buffer();
        switch (dtype) {
            case NDArray.DType.INT8, NDArray.DType.UINT8 -> {
                var bb = (ByteBuffer)inBuffer;
                while (bb.hasRemaining()) {
                    var pixel = bb.get();
                    buffer.put(pixel);
                }
            }
            case NDArray.DType.INT16, NDArray.DType.UINT16 -> {
                var bb = (ShortBuffer)inBuffer;
                while (bb.hasRemaining()) {
                    var pixel = bb.get();
                    buffer.putShort(pixel);
                }
            }
            case NDArray.DType.INT32 -> {
                var bb = (IntBuffer)inBuffer;
                while (bb.hasRemaining()) {
                    var pixel = bb.get();
                    buffer.putInt(pixel);
                }
            }
            case NDArray.DType.FLOAT32 -> {
                var bb = (FloatBuffer)inBuffer;
                while (bb.hasRemaining()) {
                    var pixel = bb.get();
                    buffer.putFloat(pixel);
                }
            }
            case NDArray.DType.FLOAT64 -> {
                var bb = (DoubleBuffer)inBuffer;
                while (bb.hasRemaining()) {
                    var pixel = bb.get();
                    buffer.putDouble(pixel);
                }
            }
        }
        return ndArray;
    }

    public static Mat ndArrayToMat(NDArray nd) {
        int[] shape = nd.shape().toIntArray();
        if (shape.length != 4 || shape[0] != 1) {
            throw new IllegalArgumentException("Expected shape [1, C, H, W], got: " + java.util.Arrays.toString(shape));
        }

        int c = shape[1];
        int h = shape[2];
        int w = shape[3];

        ByteBuffer byteBuffer = nd.buffer().order(ByteOrder.nativeOrder());
        ShortBuffer shortBuffer = byteBuffer.asShortBuffer();
        if (shortBuffer.remaining() != c * h * w) {
            throw new IllegalStateException("NDArray buffer size mismatch.");
        }

        short[] planar = new short[c * h * w];
        shortBuffer.get(planar);

        short[] interleaved = new short[h * w * c];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                for (int ch = 0; ch < c; ch++) {
                    int planarIndex = ch * h * w + y * w + x;
                    int interleavedIndex = (y * w + x) * c + ch;
                    interleaved[interleavedIndex] = planar[planarIndex];
                }
            }
        }
        ShortPointer shortPointer = new ShortPointer(interleaved);

        return new Mat(h, w, CvType.CV_16UC(c), shortPointer);
    }


    private static void runPythonCode() throws IOException, InterruptedException, URISyntaxException, BuildException, TaskException {
        var server =  QPEx.getCurrentServer();
        var selectedObject = QPEx.getSelectedObject();
        if (selectedObject == null || server == null) {
            return;
        }
        selectedObject.setLocked(true);
        var biServer = (ImageServer<BufferedImage>)server;
        var regionRequest = RegionRequest.createInstance(QPEx.getCurrentServer().getPath(), 1, selectedObject.getROI());
        var ndArray = NDArrays.fromBufferedImage(biServer, regionRequest);
        var pxSize = QPEx.getCurrentImageData().getServerMetadata().getAveragedPixelSize();
        Map<String, Object> inputs = Map.of("ndarray", ndArray, "pixel_size", pxSize);
        String model = QPEx.getCurrentImageData().isBrightfield() ? "brightfield_nuclei" : "fluorescence_nuclei_and_cells";
        String pythonScript = String.format("""
from instanseg import InstanSeg
from appose import NDArray
import numpy as np

ins = InstanSeg("%s")
# inputs["ndarray"] is automagically assigned in the global python environment
ins_output = ins.eval_medium_image(ndarray.ndarray(), return_image_tensor=False, pixel_size=pixel_size).numpy()

# appose's shared memory array from the python side
nda = NDArray(dtype="uint16", shape=ins_output.shape)
# create a np.ndarray-like view on the ndarray buffer
nda_np = nda.ndarray()
# copy the instanseg output into the ndarray buffer
nda_np[:] = ins_output
# return the ndarray view on the shared memory
task.outputs["instanseg_output"] = nda
""", model);
        String envContent;
        try (InputStream inputStream = NDArrays.class.getResourceAsStream("instanseg.yml")) {
            try (InputStreamReader isr = new InputStreamReader(inputStream);
                 BufferedReader reader = new BufferedReader(isr)) {
                envContent = reader.lines().collect(Collectors.joining(System.lineSeparator()));
            }
        }
        Environment env = Appose.mamba().channels("conda-forge").content(envContent).build();
        try (Service python = env.python().init("import numpy")) {
            Service.Task task = python.task(pythonScript, inputs);
            task.listen(event -> {
                switch (event.responseType) {
                    case UPDATE:
                        logger.info("Progress: " + event.current + "/" + event.maximum);
                        break;
                    case COMPLETION:
                        logger.info("Task complete. Result: " + task.outputs);
                        var isOutput = (NDArray) task.outputs.get("instanseg_output");
                        // todo: general method to convert back to mat or BI
                        Mat mat = NDArrays.ndArrayToMat(isOutput);
                        OutputHandler.OutputToObjectConverter<Mat, Mat, Mat[]> outputHandler = new InstanSegOutputToObjectConverter(
                                List.of(),
                                PathCellObject.class
                        );
                        var handler = OutputHandler.createObjectOutputHandler(outputHandler);
                        Parameters.Builder<Mat, Mat> builder = Parameters.builder();

                        var params = builder.imageData(QPEx.getCurrentImageData())
                                .server((ImageServer<BufferedImage>) QPEx.getCurrentServer())
                                .imageFunction( (parameters) -> {
                                    return OpenCVTools.imageToMat(parameters.getServer().readRegion(parameters.getRegionRequest()));
                                })
                                .parent(selectedObject)
                                .region(regionRequest)
                                .build();
                        handler.handleOutput(params, new Mat[] {mat});
                        Objects.requireNonNull(QPEx.getCurrentHierarchy()).fireHierarchyChangedEvent(null);
                        break;
                    case CANCELATION:
                        logger.info("Task canceled");
                        break;
                    case FAILURE:
                        logger.error("Task failed: " + task.error);
                        break;
                    case CRASH:
                        logger.error("Task crashed:" + task.error);
                }
            });
            task.start();
            Thread.sleep(1000);
            if (!task.status.isFinished()) {
                // Task is taking too long; request a cancelation.
                task.cancel();
            }
            task.waitFor();
        }
    }

}
