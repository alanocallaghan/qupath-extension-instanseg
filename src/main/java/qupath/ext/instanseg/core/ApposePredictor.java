package qupath.ext.instanseg.core;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import org.apposed.appose.Appose;
import org.apposed.appose.BuildException;
import org.apposed.appose.Environment;
import org.apposed.appose.NDArray;
import org.apposed.appose.Service;
import org.apposed.appose.TaskException;
import org.bytedeco.opencv.opencv_core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ApposePredictor {
    private static final Logger logger = LoggerFactory.getLogger(ApposePredictor.class);
    private Service python;

    private ApposePredictor(Path modelPath, String device) throws BuildException, IOException, TaskException, InterruptedException {
        logger.info("Creating environment");
        Environment env = Appose.pixi(NDArrays.class.getResource("pixi.toml")).build();
        String pythonScript = String.format("""
from instanseg import InstanSeg
from appose import NDArray
import torch
import numpy as np

model = torch.jit.load("%s")
ins = InstanSeg(model, device="%s")
""", modelPath.toString(), device);
        logger.info("Starting task");
        python = env.python().init(pythonScript);
        python.task("pass").waitFor();
    }

    static ApposePredictor create(Path modelPath, String device) throws BuildException, IOException, TaskException, InterruptedException {
        return new ApposePredictor(modelPath, device);
    }

    CompletableFuture<Mat> predict(Mat mat, double pixelSize) {

        String script = """
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
                """;

        var nda = NDArrays.matToNDArray(mat);

        Map<String, Object> inputs = Map.of("ndarray", nda, "pixel_size", pixelSize);
        CompletableFuture<Mat> future = new CompletableFuture<>();

        Service.Task task = python.task(script, inputs);
        task.listen(event -> {
            switch (event.responseType) {
                case UPDATE:
                    logger.info("Progress: " + event.current + "/" + event.maximum);
                    break;
                case COMPLETION:
                    logger.info("Task complete. Result: " + task.outputs);
                    var isOutput = (NDArray) task.outputs.get("instanseg_output");
                    // todo: general method to convert back to mat or BI
                    Mat matOutput = NDArrays.ndArrayToMat(isOutput);
                    future.complete(matOutput);
                    break;
                case CANCELATION:
                    logger.info("Task canceled");
                    // todo exceptionally? and below
                    future.complete(null);
                    break;
                case FAILURE:
                    logger.error("Task failed: " + task.error);
                    future.complete(null);
                    break;
                case CRASH:
                    logger.error("Task crashed:" + task.error);
                    future.complete(null);
                    break;
            }
        });
        task.start();
        return future;
    }

    public void close() {
        python.close();
    }
}
