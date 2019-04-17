package com.test.ml.commons;

import java.util.ArrayList;
import java.util.Map;

import com.test.ml.commons.data.model.ModelParams;
import com.test.ml.commons.dtype.Dataset;
import com.test.ml.commons.dtype.DoubleMatrix;
import com.test.ml.commons.dtype.MatrixUtil;

/**
 * 
 * @author vivek
 *
 */
public class ModelUtil {

	public static void main(String[] args) {
		
		String path = args[0];
		
		CSVReader reader = new CSVReader(path, "target", ",", new ArrayList<>());
		Map<Dataset.Type, Dataset> datasets = reader.buildDataSet(false);
		Dataset train = datasets.get(Dataset.Type.train);
		
		ModelUtil util = new ModelUtil();
		util.fitModel(train.getFeatures(), train.getLabel(), 
				null, null, 100000, 0.001);

	}
	
	/**
	 * 
	 * @param xTrain
	 * @param yTrain
	 * @param xTest
	 * @param yTest
	 * @param numIterations
	 * @param learningRate
	 */
	public void fitModel(DoubleMatrix xTrain, DoubleMatrix yTrain, 
			DoubleMatrix xTest, DoubleMatrix yTest, 
			int numIterations, double learningRate) {
		DoubleMatrix weights = new DoubleMatrix(1,xTrain.getCols());
		weights.initWithZero();
		double bias = 0.0;
		
		PropogateUtil propUtil = new PropogateUtil();
		
		ModelParams modelParams = propUtil.optimize(weights, bias, 
				xTrain, yTrain, numIterations, learningRate);
		
		double threshold = 0.85;
		
		DoubleMatrix trainPreds = predict(modelParams, xTrain, threshold);
		double trainAccrPerc = computeAccuracy(yTrain, trainPreds)*100;
		System.out.printf("Prediction accuracy on train is: %f\n", trainAccrPerc );
		
		
		if(xTest != null) {
			DoubleMatrix testPreds = predict(modelParams, xTest, threshold);
			double testAccrPerc = computeAccuracy(yTest, testPreds)*100;
			System.out.printf("Prediction accuracy on test is: %f\n", testAccrPerc );
		}
		
	}
	
	public double computeAccuracy(DoubleMatrix Y, DoubleMatrix preds) {
		System.out.println("Actual\tPreds");
		for(int i=0;i<Y.getRows();i++)
		System.out.println(Y.get(i, 0) +"\t"+ preds.get(i, 0));
		DoubleMatrix absDiff = MatrixUtil.applyAbs(MatrixUtil.subtract(Y, preds));
		double mean = MatrixUtil.sumAll(absDiff)/absDiff.getRows();
		
		return mean;
	}
	
	/**
	 * 
	 * @param w
	 * @param b
	 * @param X
	 * @return
	 */
	public DoubleMatrix predict(ModelParams model, DoubleMatrix X,
			double threshold) {
		
		DoubleMatrix w = model.getWeights();
		double b = model.getBias();
		
		int m = X.getRows();
		DoubleMatrix preds = new DoubleMatrix(m, 1);
		
		DoubleMatrix activation = MatrixUtil.applySigmoid(MatrixUtil.addScalar(
				MatrixUtil.multiply(X, MatrixUtil.trans(w)), b));
		
		for(int i=0;i<m;i++) {
			if(activation.getData()[i][0] <= threshold)
				preds.getData()[i][0] = 0;
			else
				preds.getData()[i][0] = 1;
		}
		
		return preds;
	}

}
