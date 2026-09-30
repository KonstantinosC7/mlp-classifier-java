// Basileios Skarafigkas, A.M : 4491
// Christopoulos Konstantinos, A.M : 4527
import java.io.FileNotFoundException;
import java.util.*;

public class MLP {
	private int d;		// number of inputs
	private int K;		// number of clusters
	private int H1;		// number of neuron in the first layer 
	private int H2;		// number of neuron in the second layer
	private int H3;		// number of neuron in the third layer
	private String activationFunction;
	private float[][] trainInputs;
	private float[][] trainCategory;
	private float[][] testInputs;
	private float[][] testCategory;
	private int[] yPred;				// ta ypred gia plot
	private double learningRate;
	private float threshold;
	private float[][] W1, W2, W3, W4; 	// W4 for output layer
	private float[] b1, b2 , b3, b4;	// b4 for output layer
	private int B;						// mini-batch 
	
	private float[] y1;					// first layer outputs
	private float[] y2;					// second layer outputs
	private float[] y3;					// third layer outputs
	private float[] y;
	
	private float[] delta4;				// error for every neuron in the output layer
    private float[] delta3;				// error for every neuron in the H3 layer
    private float[] delta2;				// error for every neuron in the H2 layer
    private float[] delta1;				// error for every neuron in the H1 layer
    
    private float[] derivativeW4;		// derivative for Ws in output layer
    private float[] derivativeW3;		// derivative for Ws in H3 layer
    private float[] derivativeW2;		// derivative for Ws in H2 layer
    private float[] derivativeW1;		// derivative for Ws in H1 layer
    
    private float[] derivativeb1;		// derivative for Ws in H1 layer
    private float[] derivativeb2;
    private float[] derivativeb3;
    private float[] derivativeb4;
	
	public MLP(int d, int K, int H1, int H2, int H3, String activationFunction, float learningRate, float threshold, int B) throws FileNotFoundException {
		
		LoadData ld = new LoadData();
		ld.loadDataFromFile();
		
		this.d = d;
		this.K = K;
		this.H1 = H1;
		this.H2 = H2;
		this.H3 = H3;
		this.activationFunction = activationFunction;
		this.trainInputs = ld.getTrainInputs();
		this.trainCategory = ld.getTrainOutputs();
		this.testInputs = ld.getTestInputs();
		this.testCategory = ld.getTestOutputs();
		this.yPred = new int[testCategory.length];
		this.learningRate = learningRate;
		this.threshold = threshold;
		this.B = B;
		
		this.W1 = new float[d][H1];
        this.b1 = new float[H1];
        createRandomWs(this.W1, this.b1, d, H1);
        this.W2 = new float[H1][H2];
        this.b2 = new float[H2];
        createRandomWs(this.W2, this.b2, H1, H2);
        this.W3 = new float[H2][H3];
        this.b3 = new float[H3];
        createRandomWs(this.W3, this.b3, H2, H3);
        this.W4 = new float[H3][K];
        this.b4 = new float[K];
        createRandomWs(this.W4, this.b4, H3, K);
		
        this.y1 = new float[H1];		// first layer outputs
        this.y2 = new float[H2];		// second layer outputs
        this.y3 = new float[H3];		// third layer outputs
        this.y = new float[K];
        
        this.delta4 = new float[K];	
        this.delta3 = new float[H3];
        this.delta2 = new float[H2];
        this.delta1 = new float[H1];
        
        this.derivativeW4 = new float[K];
        this.derivativeW3 = new float[H3];
        this.derivativeW2 = new float[H2];
        this.derivativeW1 = new float[H1];
        this.derivativeb1 = new float[H1];
        this.derivativeb2 = new float[H2];
        this.derivativeb3 = new float[H3];
        this.derivativeb4 = new float[K];
	}
	
	public int[] getYPred() {
		return this.yPred;
	}
	
	private void createRandomWs(float[][] Ws, float[] bs, int inputs, int outputs){
		for (int i = 0; i < inputs; i++) {
            for (int j = 0; j < outputs; j++) {
                Ws[i][j] = (float) (Math.random() * 2 - 1);
            }
        }
        for (int i = 0; i < outputs; i++) {
            bs[i] = (float) (Math.random() * 2 - 1);
        }
	
	}
	private float activationDerivative(float x) {
	    float derivative = 0;
	    if (activationFunction.equals("relu")) {
	        derivative = (x > 0) ? 1 : 0;
	    }
	    if (activationFunction.equals("tanh")) {
	        derivative = 1 - x * x;
	    }
	    if (activationFunction.equals("logistic")) {
	        derivative = x * (1 - x);
	    }
	    return derivative;
	}
	private float activation(float x) {
		if (activationFunction.equals("relu")) {
			x = Math.max(0, x);
		}
		if (activationFunction.equals("tanh")){
			x = (float) Math.tanh(x);
		}
		if (activationFunction.equals("logistic")) {
			x = (float) (1.0 / (1.0 + Math.exp(-x)));
		}
		return x;
	}
	public float[] softmax(float[] x) {
        float[] y = new float[x.length];
        float sum = 0;
        for (int i = 0; i < x.length; i++) {
            sum += Math.exp(x[i]);
        }
        for (int i = 0; i < x.length; i++) {
            y[i] = (float) Math.exp(x[i]) / sum;
        }
        return y;
    }
	
	public float[] derivativeSoftmax(float[] y) {
		float[] derivative = new float[y.length];
		for (int i = 0; i < y.length; i++) {
		    float sum = 0;
		    for (int j = 0; j < y.length; j++) {
		      if (i != j) {
		        sum += y[j];
		      }
		    }
		    derivative[i] = y[i] * sum;
		  }
		  return derivative;
	}
	
	private int getMaxIndex(float[] arr) {
        int index = 0;
        float max = arr[0];
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > max) {
                max = arr[i];
                index = i;
            }
        }
        return index;
    }
	
	// eisodo ena dedomeno kai briskei thn exodo
	public float[] forwardPass(float[] x, int d, float[] y, int K) {
		
		// initialize to zero
		// exodos tou kathe neurwna se kathe epipedo
		this.y1 = new float[this.H1];	// hidden layer 1
		this.y2 = new float[this.H2];	// hidden layer 2
		this.y3 = new float[this.H3];	// hidden layer 3
		this.y = new float[this.K];		// output layer
		
		
		// first hidden layer
        for (int i = 0; i < H1; i++) {
            for (int j = 0; j < d; j++) {
                y1[i] += W1[j][i] * x[j];		// synolikh eisodos tou neurwna i sto epipedo 1
            }
            y1[i] += this.b1[i];
            y1[i] = activation(y1[i]);
        }
        
        // second hidden layer
        for (int i = 0; i < H2; i++) {
            for (int j = 0; j < H1; j++) {
                y2[i] += W2[j][i] * y1[j];		// synolikh eisodos tou neurwna i sto epipedo 2
            }
            y2[i] += b2[i];
            y2[i] = activation(y2[i]);
        }
        
        // third hidden layer
        for (int i = 0; i < H3; i++) {
            for (int j = 0; j < H2; j++) {
                y3[i] += W3[j][i] * y2[j];
            }
            y3[i] += b3[i];
            y3[i] = activation(y3[i]);
        }
        // for output layer
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < H3; j++) {
                y[i] += W4[j][i] * y3[j];
            }
            y[i] += b4[i];
            
        }
        // store the output vector for the specific input 
        y = softmax(y);
        return y;
	}
	
	
	public void backProp(float[] x, int d, float[] t, int K) { 
		// initialize to zero
		this.delta1 = new float[this.H1];
		this.delta2  = new float[this.H2];
		this.delta3 = new float[this.H3];
		this.delta4 = new float[this.K];
		
		// calculate error at output layer
	    for (int j = 0; j < K; j++) {
	    	delta4[j] = y[j] - t[j];
	    }
	    float[] derivativeSoftmax = derivativeSoftmax(y);
	    for (int j = 0; j < K; j++) {
	        //delta4[j] *= logisticDerivative(y[j]);
	    	delta4[j] *= derivativeSoftmax[j];
	    }
	    
	    // calculate error at H3  hidden layer
	    for (int k = 0; k < H3; k++) {
	    	for (int j = 0; j < K; j++) {
	    		delta3[k] += W4[k][j] * delta4[j];
	        }
	    	delta3[k] *= activationDerivative(y3[k]);
	    }
	    // calculate error at H2 hidden layer
	    for (int k = 0; k < H2; k++) {
	    	for (int j = 0; j < H3; j++) {
	    		delta2[k] += W3[k][j] * delta3[j];
	        }
	    	delta2[k] *= activationDerivative(y2[k]);
	    }
	        // calculate error at H1 hidden layer
	    for (int k = 0; k < H1; k++) {
	    	for (int j = 0; j < H2; j++) {
	    		delta1[k] += W2[k][j] * delta2[j];
	        }
	        delta1[k] *= activationDerivative(y1[k]);
	    } 
	    
	    
	    // calculate derivatives for output layer
	    // weights
	    for (int k = 0; k < H3; k++) {
	    	for (int j = 0; j < K; j++) {
	    		derivativeW4[j] += delta4[j] * y3[k];
	        }
	    }
	    // biases
	    for(int i = 0; i < K; i ++) {
	    	derivativeb4[i] += delta4[i];
	    }
	        
	        
	    // calculate derivative for H3 layer
	    for (int k = 0; k < H2; k++) {
	    	for (int j = 0; j < H3; j++) {
	    		derivativeW3[j] += delta3[j] * y2[k];
	        }
	    }
	    for(int i = 0; i < H3; i ++) {
	    	derivativeb3[i] += delta3[i];
	    }
	        
	    // calculate derivative for H2 layer
	    for (int k = 0; k < H1; k++) {
	    	for (int j = 0; j < H2; j++) {
	          	derivativeW2[j] += delta2[j] * y1[k];
	        }
	     }
	     for(int i = 0; i < H2; i ++) {
	       	derivativeb2[i] += delta2[i];
	     }
	        
	     // calculate derivative for H1 layer
	     for (int k = 0; k < d; k++) {
	    	 for (int j = 0; j < H1; j++) {
	          	derivativeW1[j] += delta1[j] * x[k];
	         }
	     }
	     for(int i = 0; i < H1; i ++) {
	       	derivativeb1[i] += delta1[i];
	     }
		
	}
	
	public void gradientDescent() {
		// update weights and biases for output layer
        for (int i = 0; i < H3; i++) {
            for (int j = 0; j < K; j++) {
                this.W4[i][j] -= learningRate * derivativeW4[j];
            }
        }
        // biases
        for (int i = 0; i < K; i++) {
        	this.b4[i] -= learningRate * derivativeb4[i];
        }
        
        // update weights and biases for H3 layer
        for (int i = 0; i < H2; i++) {
            for (int j = 0; j < H3; j++) {
                this.W3[i][j] -= learningRate * derivativeW3[j];
            }	
        }
        // biases
        for (int i = 0; i < H3; i++) {
        	this.b3[i] -= learningRate * derivativeb3[i];
        }
        
        // update weights and biases for H2 layer
        for (int i = 0; i < H1; i++) {
            for (int j = 0; j < H2; j++) {
                this.W2[i][j] -= learningRate * derivativeW2[j];
            } 
        }
        // biases
        for (int i = 0; i < H2; i++) {
        	this.b2[i] -= learningRate * derivativeb2[i];
        }
        // update weights and biases for H1 layer
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < H1; j++) {
                this.W1[i][j] -= learningRate * derivativeW1[j];
            }
        }
        for (int i = 0; i < H1; i++) {
        	this.b1[i] -= learningRate * derivativeb1[i];
        }
	}
	
	// ypologizei to error kathe paradeigmatos (E^n)
	private double errorComputation(float[] predictedY, float[] realY) {
		double error = 0;
		for(int i = 0; i < predictedY.length; i ++) {
			error +=  Math.pow((predictedY[i] - realY[i]), 2);
		}
		return 0.5*error;
		//return error;
	}
	
	public int trainMLP() {
		int N = trainInputs.length;
		int miniBatches = N/B;			// plhthos omades paradeigamtwn
		double previousEr; 
		Double currentEr = Double.MAX_VALUE;
		int epoch = 0;					// epoxes
		boolean anotherEpoch = false;
		float[][] xTrain;
		float[][] yTrain;
		while(epoch < 700 || anotherEpoch == true) {
			// epeidh to xwrizoume se mini batches
			previousEr = currentEr;
			currentEr = 0.0;
			
			// gia kathe paradeigma pou anhkei sthn omada mhkous mini-batch
			for(int i = 0; i < miniBatches; i ++) {
				// apo i*B mexri (i+1)*B -> kathe fora
				xTrain = Arrays.copyOfRange(this.trainInputs, i * B, (i + 1) * B);
	            yTrain = Arrays.copyOfRange(this.trainCategory, i * B, (i + 1) * B);
	            // initialize derivatives to zero for this mini-batch
	            this.derivativeW1 = new float[this.H1];
				this.derivativeW2 = new float[this.H2];
				this.derivativeW3 = new float[this.H3];
				this.derivativeW4 = new float[this.K];
				this.derivativeb1 = new float[H1];
		        this.derivativeb2 = new float[H2];
		        this.derivativeb3 = new float[H3];
		        this.derivativeb4 = new float[K];
				
	            for(int j = 0; j < B; j ++) {			// gia kathe paradeigma ths omadas
	            	// forward pass and we store the predicted output in y
	            	this.y = forwardPass(xTrain[j], this.d, this.y, this.K);
	                currentEr += errorComputation(this.y, yTrain[j]);
	                // back propagation : calculates derivatives for weights and biases 
	                backProp(xTrain[j], d, yTrain[j], K);
	            }
	            gradientDescent();	           
			}			
			epoch += 1;
			//System.out.println("For epoch " + epoch + ", error is : " + (double)currentEr/trainInputs.length);
			
			// check if currentEr/trainInputs.length is NaN and restart  
			if(Double.isNaN(currentEr/trainInputs.length)) {
				return -1;
			}
			// gia na synexxizei na trexei meta tis 700 epoxes
			if(epoch == 700) {
				anotherEpoch = true;
			}
			if(previousEr-currentEr < this.threshold && anotherEpoch == true) {
				break;
			}
		}
		System.out.println("MLP terminated at Epoch " + epoch + ", error is : " + (double)currentEr/trainInputs.length);
		return 0;
	}
	
	private boolean isSameCategory(float[] yPred, float[] yReal) {
		float indexPred = getMaxIndex(yPred);			
		float indexReal = getMaxIndex(yReal);			
		return indexPred == indexReal;
	}
	
	
	public float testMLP() {
		int correct = 0;
		for(int i = 0; i < this.testInputs.length; i ++) {		// pairnoume kathe dedomeno
			this.y = forwardPass(this.testInputs[i], d, y, K);	// ypologizoume thn exodo tou
			//System.out.println(y[0] + ", " + y[1] + ", "+ y[2]);
			if(isSameCategory(this.y, this.testCategory[i])) {	// elegxoume an einai idias kathgorias
				correct += 1;
				yPred[i] = 1;									// 1 means the same
			}
			else{
				yPred[i] = 0;
			}
		}
		return (float)correct/this.testInputs.length;
	}
	
}	