package mapconstruction2;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/** Supply trajectories in the manuscript run's recorded order. */
public class Chicago_Ahmed_Runner {
    public static void main(String[] args) throws Exception {
        List<PoseFile> trajectories = new ArrayList<PoseFile>();
        boolean hasAltitude = Boolean.parseBoolean(args[3]);
        for (String filename : Files.readAllLines(Paths.get(args[5]), StandardCharsets.UTF_8)) {
            trajectories.add(PoseFile.readFile(new File(args[0], filename), hasAltitude));
        }
        MapConstruction method = new MapConstruction();
        List<Vertex> network = method.constructMapMain(
            trajectories, Double.parseDouble(args[2]), Double.parseDouble(args[4]));
        MapConstruction.writeToFile(network, args[1]);
    }
}
