// Basileios Skarafigkas, A.M : 4491
// Christopoulos Konstantinos, A.M : 4527
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

public class Classification {
	static float[][] trainingSamples = new float[4000][2];
    static float[][] testSamples = new float[4000][2];
    static String[] trainingLabels = new String[4000];
    static String[] testLabels = new String[4000];
	
    public static void main(String[] args) {
	// Create a random number generator
		Random rand = new Random();
	// Create two arrays to store the training and test samples
	    
	
	    // Generate the training and test samples
	    for (int i = 0; i < 8000; i++) {
	        // Generate a random x1 and x2 value within the range [-1,1]
	        float x1 = rand.nextFloat() * 2 - 1;
	        float x2 = rand.nextFloat() * 2 - 1;
	
	        // Assign the sample to the appropriate array
	        if (i < 4000) {
	            trainingSamples[i][0] = x1;
	            trainingSamples[i][1] = x2;
	        } else {
	            testSamples[i - 4000][0] = x1;
	            testSamples[i - 4000][1] = x2;
	        }
	    }
	
	    // Create two arrays to store the training and test labels
	    
	
	    // Classify the training and test samples
	    for (int i = 0; i < 8000; i++) {
	        float x1, x2;
	        String label;
	        if (i < 4000) {
	            x1 = trainingSamples[i][0];
	            x2 = trainingSamples[i][1];
	            label = classify(x1, x2);
	            trainingLabels[i] = label;
	        } else {
	            x1 = testSamples[i - 4000][0];
	            x2 = testSamples[i - 4000][1];
	            label = classify(x1, x2);
	            testLabels[i - 4000] = label;
	        }
	    }
	    writeToFile(trainingSamples, trainingLabels, "training.csv");
	    writeToFile(testSamples, testLabels, "test.csv");
	}


	public static String classify(float x1, float x2) {
	    // Check the first condition
		// 1. C1 : (x1 – 0.5)^2 + (x2 – 0.5)^2 <0.2 && x2>0.5
	    if (((x1 - 0.5) * (x1 - 0.5) + (x2 - 0.5) * (x2 - 0.5)) < 0.2 && x2 > 0.5) {
	        return "C1";
	    }
	
	    // Check the second condition
	    // 2. C2 : (x1 – 0.5)^2 + (x2 – 0.5)^2 <0.2 && x2<0.5
	    else if (((x1 - 0.5) * (x1 - 0.5) + (x2 - 0.5) * (x2 - 0.5)) < 0.2 && x2 < 0.5) {
	        return "C2";
	    }
	
	    // Check the third condition
	    // 3. C1 : (x1 + 0.5)^2 + (x2 + 0.5)^2 <0.2 && x2>-0.5
	    else if (((x1 + 0.5) * (x1 + 0.5) + (x2 + 0.5) * (x2 + 0.5)) < 0.2 && x2 > -0.5) {
	        return "C1";
	    }
	
	    // Check the fourth condition
	    // 4. C2 : (x1 + 0.5)^2 + (x2 + 0.5)^2 <0.2 && x2<-0.5
	    else if (((x1 + 0.5) * (x1 + 0.5) + (x2 + 0.5) * (x2 + 0.5)) < 0.2 && x2 < - 0.5) {
	        return "C2";
	    }
	    // Check the fifth condition
	    // 5. C1 : (x1 - 0.5)^2 + (x2 + 0.5)^2 <0.2 && x2>-0.5
	    else if (((x1 - 0.5) * (x1 - 0.5) + (x2 + 0.5) * (x2 + 0.5)) < 0.2 && x2 > -0.5) {
	        return "C1";
	    }
	
	    // Check the sixth condition
	    // 6. C2 : (x1 - 0.5)^2 + (x2 + 0.5)^2 <0.2 && x2<-0.5
	    else if (((x1 - 0.5) * (x1 - 0.5) + (x2 + 0.5) * (x2 + 0.5)) < 0.2 && x2 < -0.5) {
	        return "C2";
	    }
	
	    // Check the seventh condition
	    // 7. C1 : (x1 + 0.5)^2 + (x2 - 0.5)^2 <0.2 && x2>0.5
	    else if (((x1 + 0.5) * (x1 + 0.5) + (x2 - 0.5) * (x2 - 0.5)) < 0.2 && x2 > 0.5) {
	        return "C1";
	    }
	
	    // Check the eighth condition
	    // 8. C2 : (x1 + 0.5)^2 + (x2 - 0.5)^2 <0.2 && x2<0.5
	    else if (((x1 + 0.5) * (x1 + 0.5) + (x2 - 0.5) * (x2 - 0.5)) < 0.2 && x2 < 0.5) {
	        return "C2";
	    }
	    else {
	    	return "C3";
	    }

	}

	public static void writeToFile(float[][] samples, String[] labels, String fileName) {
	    try {
	        // Create a BufferedWriter to write to the file
	        BufferedWriter writer = new BufferedWriter(new FileWriter(fileName));
	
	        // Write each sample and label to the file
	        for (int i = 0; i < samples.length; i++) {
	            // Write the x1 and x2 values for the sample
	            writer.write(String.format("%f %f ", samples[i][0], samples[i][1]));
	            // Write the label for the sample
	            writer.write(labels[i]);
	            // Add a new line after each sample
	            writer.newLine();
	        }
	
	        // Close the BufferedWriter
	        writer.close();
	    } catch (IOException e) {
	        // Print an error message if there is a problem writing to the file
	        System.out.println("Error writing to file " + fileName);
	    }
	}	
}
	
   
