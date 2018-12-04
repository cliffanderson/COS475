import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class MakeImages {

    public static void main(String[] args) throws Exception {
        File origFile = new File("../orig.txt");
        File newDigitFile = new File("../new.txt");
        File origFullResFile = new File("../orig_full_res.txt");

        List<String> origLines = new ArrayList<>();
        List<String> newDigitLines = new ArrayList<>();
        List<String> origFullResLines = new ArrayList<>();

        BufferedReader in = new BufferedReader(new FileReader(origFile));
        while(in.ready()) {
            origLines.add(in.readLine());
        }

        in = new BufferedReader(new FileReader(newDigitFile));
        while(in.ready()) {
            newDigitLines.add(in.readLine());
        }

        in = new BufferedReader(new FileReader(origFullResFile));
        while(in.ready()) {
            origFullResLines.add(in.readLine());
        }

        in.close();



        List<Double> origPixels = new ArrayList<>();
        List<Double> newDigitPixels = new ArrayList<>();
        List<Double> origFullResPixels = new ArrayList<>();

        for(String s : origLines) {
            //System.out.println(s);
            double d = Double.parseDouble(s);
            origPixels.add(d);
        }

        for(String s : newDigitLines) {
            double d = Double.parseDouble(s);
            newDigitPixels.add(d);
        }

        for(String s :  origFullResLines) {
            double d = Double.parseDouble(s);
            origFullResPixels.add(d);
        }

        System.out.println("Orig pixels: " + origPixels);
        //System.out.println(newDigitPixels);


        BufferedImage origImage = new BufferedImage(14, 14, BufferedImage.TYPE_INT_RGB);
        for(int x = 0; x < 14; x++) {
            for(int y = 0; y < 14; y++) {
                int rgb = origPixels.get(y*14 + x).intValue();
                //System.out.println(rgb);
                origImage.setRGB(x, y, new Color(rgb, rgb, rgb).getRGB());
            }
        }

        ImageIO.write(origImage, "PNG", new File("orig.png"));



        BufferedImage newImage = new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB);
        for(int x = 0; x < 28; x++) {
            for(int y = 0; y < 28; y++) {
                int rgb = newDigitPixels.get(y*28 + x).intValue();
                //System.out.println(rgb);
                newImage.setRGB(x, y, new Color(rgb, rgb, rgb).getRGB());
            }
        }

        ImageIO.write(newImage, "PNG", new File("newImage.png"));



        BufferedImage origFullResImage = new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB);
        for(int x = 0; x < 28; x++) {
            for(int y = 0; y < 28; y++) {
                int rgb = origFullResPixels.get(y*28 + x).intValue();
                //System.out.println(rgb);
               origFullResImage.setRGB(x, y, new Color(rgb, rgb, rgb).getRGB());
            }
        }

        ImageIO.write(origFullResImage, "PNG", new File("origFullRes.png"));


    }
}