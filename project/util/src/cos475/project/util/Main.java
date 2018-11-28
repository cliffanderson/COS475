package cos475.project.util;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Map;

public class Main {

    public static void main(String[] args) throws Exception {
        // Map of labels
        // Key is image file name (minus the .png) and value is the label (0-9)
        Map<Integer, Integer> labels = new HashMap<Integer, Integer>();

        for (int i = 0; i < 10; i++) {
            File sourceDir = new File("/Users/cliffanderson/Downloads/mnist_png/training/" + i + "/");
            File destDir = new File("/Users/cliffanderson/Dropbox/Fall2018/Machine Learning/COS475/project/mnist_digit_dataset_training/" + i);

            destDir.mkdir();

            for (File f : sourceDir.listFiles()) {
                BufferedImage image = ImageIO.read(f);
                BufferedImage newImage = new BufferedImage(14, 14, BufferedImage.TYPE_INT_RGB);
                System.out.println("Wrote " + f.getName());

                // Entry for labels map
                labels.put(Integer.parseInt(f.getName().replace(".png", "")), i);

                for (int x = 0; x < 14; x++) {
                    for (int y = 0; y < 14; y++) {
                        int average = new Color(image.getRGB(x * 2, y * 2)).getBlue() + new Color(image.getRGB(x * 2, y * 2 + 1)).getBlue() +
                                new Color(image.getRGB(x * 2 + 1, y * 2)).getBlue() + new Color(image.getRGB(x * 2 + 1, y * 2 + 1)).getBlue();

                        average /= 4;
                        newImage.setRGB(x, y, new Color(average, average, average).getRGB());

                    }
                }

                File imageFile = new File(destDir, f.getName());
                if (imageFile.exists()) {
                    imageFile.delete();
                }

                ImageIO.write(newImage, "PNG", imageFile);
            }
        }

        // write labels to file
        File labelFile = new File("/Users/cliffanderson/Dropbox/Fall2018/Machine Learning/COS475/project/mnist_digit_dataset_training_labels.csv");
        PrintWriter out = new PrintWriter(new FileWriter(labelFile));
        for(Integer key : labels.keySet()) {
            out.println(key + "," + labels.get(key));
        }

        out.flush();
        out.close();
    }
}
