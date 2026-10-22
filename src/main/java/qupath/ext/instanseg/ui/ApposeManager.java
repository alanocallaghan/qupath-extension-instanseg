package qupath.ext.instanseg.ui;

import java.util.Collection;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import org.apposed.appose.Appose;
import org.apposed.appose.BuildException;
import org.apposed.appose.Environment;
import org.apposed.appose.TaskException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.instanseg.core.NDArrays;

public class ApposeManager {
    private static final Logger logger = LoggerFactory.getLogger(ApposeManager.class);
    private boolean initialized;
    private Environment env;

    public boolean isInitialized() {
        return initialized;
    }

    public void initialize() throws BuildException {
        logger.info("Creating environment");
        env = Appose.pixi(NDArrays.class.getResource("pixi.toml")).build();
        initialized = true;
    }

    public Collection<String> listAvailableDevices() {
        CompletableFuture<List<String>> future = new CompletableFuture<>();
        if (!initialized) {
            try {
                initialize();
            } catch (BuildException e) {
                logger.error("Error while initializing ApposeManager", e);
                return Set.of("cpu");
            }
        }
        try (var service = env.python()) {
            var task = service.task("""
                import torch
                devices = ["cpu"]
                # Check for NVIDIA CUDA GPUs
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        devices.append(f"cuda:{i}")
            
                # Check for Apple Silicon MPS (Metal Performance Shaders)
                if torch.backends.mps.is_available():
                    devices.append("mps")
            
                task.outputs["devices"] = devices
                """);
            task.listen(e -> {
                switch (e.responseType) {
                    case COMPLETION:
                        logger.info("Task complete. Result: " + task.outputs);
                        var isOutput = (List<String>)task.outputs.get("devices");
                        future.complete(isOutput);
                        break;
                    case CANCELATION:
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
            task.waitFor();
            return future.join();
        } catch (InterruptedException | TaskException e) {
            logger.error("Unable to list available devices", e);
            return Set.of("cpu");
        }
    }
}
