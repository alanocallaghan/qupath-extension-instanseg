package qupath.ext.instanseg.core;

import java.io.File;
import java.net.MalformedURLException;
import java.net.URI;
import java.net.URL;
import java.nio.file.Path;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import org.apposed.appose.Appose;
import org.apposed.appose.BuildException;
import org.apposed.appose.Environment;
import org.apposed.appose.TaskException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ApposeManager {
    private static final Logger logger = LoggerFactory.getLogger(ApposeManager.class);

    private ApposeManager() {}

    public static CompletableFuture<Environment> buildEnvironment(File file) {
        return CompletableFuture.supplyAsync(() -> {
                    try {
                        return Appose.pixi(file).build();
                    } catch (BuildException e) {
                        logger.error("Error building environment", e);
                        return null;
                    }
                }
        );
    }


    public static CompletableFuture<Environment> buildEnvironment(URL url) {
        return CompletableFuture.supplyAsync(() -> {
                    try {
                        return Appose.pixi(url).build();
                    } catch (BuildException e) {
                        logger.error("Error building environment", e);
                        return null;
                    }
                }
        );
    }

    public static CompletableFuture<Environment> buildEnvironment(Path path) {
        return buildEnvironment(path.toFile());
    }

    public static CompletableFuture<Environment> buildEnvironment(String url) {
        try {
            return buildEnvironment(URL.of(URI.create(url), null));
        } catch (MalformedURLException e) {
            throw new RuntimeException(e);
        }
    }

    public static CompletableFuture<Collection<String>> listAvailableDevices() {
        // todo should the env be a constant or field?
        return buildEnvironment(ApposePredictor.class.getResource("pixi.toml")).thenCompose(environment -> listAvailableDevicesImpl(environment));
    }

    private static CompletableFuture<Collection<String>> listAvailableDevicesImpl(Environment environment) {
        CompletableFuture<Collection<String>> future = new CompletableFuture<>();
        String initScript = """
                import torch
                import numpy
                """;
        try (var service = environment.python().init(initScript)) {
            var task = service.task("""
            import torch
            import numpy
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
                        future.complete((List<String>)task.outputs.get("devices"));
                        break;
                    case CANCELATION:
                        logger.error("Task cancelled: " + task.error);
                        future.complete(null);
                        break;
                    case FAILURE:
                        logger.error("Task failed: " + task.error);
                        future.complete(null);
                        break;
                    case CRASH:
                        logger.error("Task crashed: " + task.error);
                        future.complete(null);
                        break;
                }
            });
            task.waitFor();
        } catch (TaskException | InterruptedException e) {
            throw new RuntimeException(e);
        }
        return future;
    }
}
