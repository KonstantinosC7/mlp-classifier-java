// Basileios Skarafigkas, A.M : 4491
// Christopoulos Konstantinos, A.M : 4527
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;

public class Main {
	
	private static void storePreds(int[] ypred) {
		try (FileWriter file = new FileWriter("ypred.txt")) {
			for(int ce : ypred) {
				file.write(String.format("%d\n", ce));
			}
		} catch (IOException e) {
			System.out.println("An error occurred.");
			e.printStackTrace();
		}
	}
	
	public static void main(String[] args) throws FileNotFoundException {
		// python for plot 
		ProcessBuilder pb = new ProcessBuilder("python", "plot.py", "arg1", "arg2");
		
		// For Experiments
		int[][] numberOfHiddenNeurons = {{5,5,5}, {10,7,4}, {15,15,5}, {20,15,7},{28,28,30}};
		String[] ActivationFunctions = {"logistic", "tanh", "relu"};
		Integer[] B = {40, 400};
		// store best parameters, ypred and best score
		String bestAF = "";
		int bestB = 0;
		int[] bestHs = {-1,-1,-1};
		float bestScore = -1;
		int[] bestPred = {};
		
		for(int k = 0; k < numberOfHiddenNeurons.length; k ++) {
			for(int i = 0; i <ActivationFunctions.length; i ++) {
				for(int j = 0; j < B.length; j ++) {
					System.out.println("Running for : {H1,H2,H3} = {"+numberOfHiddenNeurons[k][0]+","+numberOfHiddenNeurons[k][1]+","+numberOfHiddenNeurons[k][2]+"}, Activation Function = " + ActivationFunctions[i] + " and B = "+ B[j]);
					// d, K, H1, H2, H3, activation function, learning rate, threshold, B
					MLP mlp = new MLP(2, 3, numberOfHiddenNeurons[k][0], numberOfHiddenNeurons[k][1], numberOfHiddenNeurons[k][2], ActivationFunctions[i], (float)0.001, (float)0.00001, B[j]);
					int ok = mlp.trainMLP();
					while(ok == -1) {		// se periptwsh NaN
						System.out.println("Retry with lower learning rate..");
						mlp = new MLP(2, 3, numberOfHiddenNeurons[k][0], numberOfHiddenNeurons[k][1], numberOfHiddenNeurons[k][2], ActivationFunctions[i], (float)0.0001, (float)0.00001, B[j]);
						ok = mlp.trainMLP();	
					}
					float currentScore = mlp.testMLP();
					System.out.println("Ability of Generalization = " + currentScore + "\n");
					if(currentScore>bestScore) {
						bestScore = currentScore;
						bestAF = ActivationFunctions[i];
						bestB = B[j];
						bestHs = numberOfHiddenNeurons[k];
						bestPred = mlp.getYPred();		// store ypred index for plot
					}
				}
				
			}
			
		}
		System.out.println("Best score is " + bestScore + ", with {H1,H2,H3} = {"+bestHs[0]+", "+bestHs[1]+", "+bestHs[2]+"}, Activation Function = " + bestAF + " and B = "+ bestB);
		storePreds(bestPred);

		try {
			pb.start();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
