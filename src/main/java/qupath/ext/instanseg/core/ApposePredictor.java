package qupath.ext.instanseg.core;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import org.apposed.appose.BuildException;
import org.apposed.appose.NDArray;
import org.apposed.appose.Service;
import org.apposed.appose.TaskException;
import org.bytedeco.opencv.opencv_core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ApposePredictor {
    private static final Logger logger = LoggerFactory.getLogger(ApposePredictor.class);
    CompletableFuture<Service> service;

    private ApposePredictor(Path modelPath, String device) {
        logger.info("Creating environment");
        // todo what if it fails? then env is a CompletableFuture<null>
        service = ApposeManager.buildEnvironment(this.getClass().getResource("pixi.toml")).thenApply(env -> {
            try (var is = this.getClass().getResourceAsStream("init-instanseg.py")) {
                assert is != null;
                String pythonScript = String.format(
                        new String(is.readAllBytes(), StandardCharsets.UTF_8),
                        modelPath.toString(), device);
                logger.info("Starting task");
                return env.python().init(pythonScript);
            } catch (IOException e) {
                logger.error("Error reading python script", e);
            }
            return null;
        });
    }

    static ApposePredictor create(Path modelPath, String device) throws BuildException, IOException, TaskException, InterruptedException {
        return new ApposePredictor(modelPath, device);
    }

    CompletableFuture<Mat> predict(Mat mat, double pixelSize) {

        String pythonScript;
        try (var is = this.getClass().getResourceAsStream("run-instanseg.py")) {
            assert is != null;
            pythonScript = new String(is.readAllBytes(), StandardCharsets.UTF_8);
            logger.debug("Starting task");
        } catch (IOException e) {
            logger.error("Error reading python script", e);
            return CompletableFuture.failedFuture(e);
        }

        var nda = NDArrays.matToNDArray(mat);

        Map<String, Object> inputs = Map.of("ndarray", nda, "pixel_size", pixelSize);
        CompletableFuture<Mat> future = new CompletableFuture<>();

        return service.thenCompose(python -> {
            try {
                python.task(pythonScript, inputs)
                    .listen(event -> {
                        switch (event.responseType) {
                            case UPDATE:
                                logger.info("Progress: " + event.current + "/" + event.maximum);
                                break;
                            case COMPLETION:
                                logger.info("Task complete. Result: " + event.task.outputs);
                                var isOutput = (NDArray) event.task.outputs.get("instanseg_output");
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
                                logger.error("Task failed: " + event.task.error);
                                future.complete(null);
                                break;
                            case CRASH:
                                logger.error("Task crashed:" + event.task.error);
                                future.complete(null);
                                break;
                        }
                    })
                    .start()
                    .waitFor();
            } catch (InterruptedException | TaskException e) {
                logger.error("Error running task", e);
                future.complete(null);
            }
            return future;
        });
    }

    public void close() {
        if (!service.isDone()) {
            service.cancel(true);
            return;
        }
        service.join().close();
    }
}
