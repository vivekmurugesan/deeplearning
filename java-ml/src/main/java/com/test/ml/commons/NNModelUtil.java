package com.test.ml.commons;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.List;
import java.util.Map;

import com.test.ml.commons.data.model.Activation;
import com.test.ml.commons.data.model.ActivationType;
import com.test.ml.commons.data.model.ModelParams;
import com.test.ml.commons.data.model.NNCache;
import com.test.ml.commons.data.model.nn.LayerParams;
import com.test.ml.commons.data.model.nn.NetworkLayout;
import com.test.ml.commons.dtype.DoubleMatrix;

/**
 * 
 * @author vivek
 *
 */
public class NNModelUtil extends ModelUtil {
	
	private String costFile = "costs_1.csv";

	public static void main(String[] args) {

	}
	
	/**
	 * 
	 * @return
	 */
	public NetworkLayout initNetwork(int layerCount, int featureCount, int classCount, 
			List<Integer> nodeCountList, long randSeed, List<ActivationType> activationTypeList) {
		int[] layerNodeCount = new int[nodeCountList.size()];
		for(int i=0;i<layerNodeCount.length;i++)
			layerNodeCount[i]=nodeCountList.get(i);
		
		Activation[] activations = new Activation[activationTypeList.size()];
		for(int i=0;i<activations.length;i++)
			activations[i] = new Activation(activationTypeList.get(i));
		
		NetworkLayout network = new NetworkLayout(layerCount, layerNodeCount, classCount, featureCount, randSeed, activations);
		network.init();
		return network;
	}
	
	/**
	 * 
	 * @param xTrain
	 * @param yTrain
	 * @param xTest
	 * @param yTest
	 * @param numIterations
	 * @param learningRate
	 * @throws FileNotFoundException 
	 */
	public void fitModel(DoubleMatrix xTrain, DoubleMatrix yTrain, 
			DoubleMatrix xTest, DoubleMatrix yTest, 
			int numIterations, double learningRate, NetworkLayout network) throws FileNotFoundException {
		
		NNPropogateUtil propUtil = new NNPropogateUtil();
		
		PrintStream ps = new PrintStream(new FileOutputStream(costFile));
		
		Map<Integer, Double> costs = propUtil.optimize(xTrain, yTrain, network, 
				 learningRate, numIterations, true, ps);
		
		double threshold = 0.5;
		
		DoubleMatrix trainPreds = predict(xTrain, yTrain, network.getLayers(), threshold);
		double trainAccrPerc = computeAccuracy(yTrain, trainPreds)*100;
		System.out.printf("Prediction accuracy on train is: %f\n", trainAccrPerc );
		
		
		if(xTest != null) {
			DoubleMatrix testPreds = predict(xTrain, yTrain, network.getLayers(), threshold);
			double testAccrPerc = computeAccuracy(yTest, testPreds)*100;
			System.out.printf("Prediction accuracy on test is: %f\n", testAccrPerc );
		}
		
	}
	
	private void printSummary(Map<Integer, Double> costs) {
		
	}


	public DoubleMatrix predict(DoubleMatrix X, DoubleMatrix Y, LayerParams[] layerParamsArr, double threshold) {

		int m = X.getRows();
		NNPropogateUtil util = new NNPropogateUtil();
		DoubleMatrix probs = util.forwardProp(X, Y, layerParamsArr);
		DoubleMatrix preds = new DoubleMatrix(m,1);

		for(int i=0;i<m;i++) {
			if(probs.get(i, 0)<= threshold)
				preds.set(i,0,0.0);
			else
				preds.set(i,0,1.0);
		}

		return preds;
	}

}
