// Basileios Skarafigkas, A.M : 4491
// Christopoulos Konstantinos, A.M : 4527
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;


public class LoadData{

	private float[][] testInputs;
	private float[][] testOutputs;
	private float[][] trainInputs;
	private float[][] trainOutputs;
	
	public LoadData(){
		this.testInputs = new float[4000][2];
		this.testOutputs = new float[4000][3];
		this.trainInputs = new float[4000][2];
		this.trainOutputs = new float[4000][3];
		
	}	
	//public static void main(String [] ARGS) throws FileNotFoundException{
	enum Category {
	    C1, C2, C3
	}
	
	public float[][] getTestInputs(){
		return this.testInputs;
	}
	
	public float[][] getTestOutputs(){
		return this.testOutputs;
	} 
	
	public float[][] getTrainInputs(){
		return this.trainInputs;
	}
	
	public float[][] getTrainOutputs(){
		return this.trainOutputs;
	}
	
	public void toString(float[][] inputs) {
		for(int i = 0; i < inputs.length; i ++) {
			for(int j = 0; j < inputs[i].length; j ++){
				System.out.print(inputs[i][j] + " ");
			}
			System.out.println();
		}
	}
	
    public void loadDataFromFile() throws FileNotFoundException {    
        String[] testOutputsLabels=new String[4000];
        Category[] testOutputCategories = new Category[4000];
        // Initialize arrays to store the outputs
        String[] trainOutputsLabels = new String[4000];
        Category[] trainOutputCategories = new Category[4000];
        // Open the CSV file for reading
        BufferedReader readerTest = new BufferedReader(new FileReader("test.csv"));
        BufferedReader readerTrain = new BufferedReader(new FileReader("training.csv"));
        
        // Read each line of the file
        String line;
        int i = 0;
        try {
			while ((line = readerTest.readLine()) != null) {
			    // Split the line on the comma
				String[] parts = line.split(" ");
				//System.out.printf(parts[0]+parts[1]+parts[2]+"\n");
			    // Parse the inputs (x1 and x2) as floats
				testInputs[i][0] = Float.parseFloat(parts[0]);
			    testInputs[i][1] = Float.parseFloat(parts[1]);

			    // Store the output (category) in the output array
			    testOutputsLabels[i] = parts[2];

			    // Increment the line counter
			    i++;
			}
		} catch (NumberFormatException e3) {
			// TODO Auto-generated catch block
			e3.printStackTrace();
		} catch (IOException e3) {
			// TODO Auto-generated catch block
			e3.printStackTrace();
		}
        // Close the reader
        try {
			readerTest.close();
		} catch (IOException e2) {
			// TODO Auto-generated catch block
			e2.printStackTrace();
		}
        i = 0;
        line = null;
        try {
			while((line = readerTrain.readLine()) != null) {
				// Split the line on the comma
				String[] parts = line.split(" ");

			    // Parse the inputs (x1 and x2) as floats
				trainInputs[i][0] = Float.parseFloat(parts[0]);
			    trainInputs[i][1] = Float.parseFloat(parts[1]);

			    // Store the output (category) in the output array
			    trainOutputsLabels[i] = parts[2];

			    // Increment the line counter
			    i++;
			}
		} catch (NumberFormatException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
        //close readeTrain
        try {
			readerTrain.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
        // Convert the outputs to floats
        
        for (i = 0; i < testOutputsLabels.length; i++) {
        	 testOutputCategories[i] = Category.valueOf(testOutputsLabels[i]);
        	 trainOutputCategories[i] = Category.valueOf(trainOutputsLabels[i]);
        }
        
        // convert the number to vector where the index of '1' is the category
        for ( i = 0; i < testOutputCategories.length; i++) {
        	if((float) testOutputCategories[i].ordinal() == 0.0) {
        		testOutputs[i][0] = (float)1.0;
        		testOutputs[i][1] = (float)0.0;
        		testOutputs[i][2] = (float)0.0;	
        	}else if((float) testOutputCategories[i].ordinal() == 1.0) {
        		testOutputs[i][0] = (float)0.0;
        		testOutputs[i][1] = (float)1.0;
        		testOutputs[i][2] = (float)0.0;
        	}else {
        		testOutputs[i][0] = (float)0.0;
        		testOutputs[i][1] = (float)0.0;
        		testOutputs[i][2] = (float)1.0;
        	}
        	
        }
        // convert the number to vector where the index of '1' is the category 
        for ( i = 0; i < trainOutputCategories.length; i++) {
        	if((float) trainOutputCategories[i].ordinal() == 0.0) {
        		trainOutputs[i][0] = (float)1.0;
        		trainOutputs[i][1] = (float)0.0;
        		trainOutputs[i][2] = (float)0.0;
        	}else if((float) trainOutputCategories[i].ordinal() == 1.0) {
        		trainOutputs[i][0] = (float)0.0;
        		trainOutputs[i][1] = (float)1.0;
        		trainOutputs[i][2] = (float)0.0;
        	}else {
        		trainOutputs[i][0] = (float)0.0;
        		trainOutputs[i][1] = (float)0.0;
        		trainOutputs[i][2] = (float)1.0;
        	}
        }
        
        
    }
}