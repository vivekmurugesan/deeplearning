package com.test.ml.commons;

import com.test.ml.commons.data.model.ForwardPropResult;
import com.test.ml.commons.data.model.ModelParams;
import com.test.ml.commons.dtype.DoubleMatrix;
import com.test.ml.commons.dtype.MatrixUtil;

/**
 * 
 * @author vivek
 *
 */
public class PropogateUtil {

	
	/**
	 * 1. Compute activation --> A. Sigmoid in this case. 
	 * A = sigmoid(MatMul(X, w.trans) + b)
	 * 2. Compute cost. 
	 * cost = (-1/m) * sumAll( Y * applyLog(A) + (1-Y) * applyLog(1-A) )
	 * @param w  column vector of dimension n x 1
	 * @param b  1
	 * @param X  matrix of dimension m x n (with m samples and n features)
	 * @param Y  column vector of dimension m x 1 (with labels for m samples)
	 */
	public ForwardPropResult forwardProp(DoubleMatrix w, double b, 
			DoubleMatrix X, DoubleMatrix Y) {
		
		int m = X.getRows(); // Sample size or number of observations.
		int n = X.getCols(); // Feature count.
		
		DoubleMatrix activation = MatrixUtil.addScalar(
				MatrixUtil.multiply(X, MatrixUtil.trans(w)), b);
		activation = MatrixUtil.applySigmoid(activation);
		
		double cost = computeCost(Y, activation, m);
		
		//System.out.println(".. cost::" + cost+ "... sum::" + sum + ".. m::" + m);
		
		return computeDerivatives(X, Y, activation, cost);
	}
	
	public double computeCost(DoubleMatrix Y, DoubleMatrix A, int m) {
		// Y * log(A)
				DoubleMatrix f1 = MatrixUtil.elementWiseMultiply(Y, MatrixUtil.applyLog(A));
				
				// (1-Y) * log(1-A)
				DoubleMatrix f2 = MatrixUtil.elementWiseMultiply(MatrixUtil.subtractFromScalar(1, Y), 
						MatrixUtil.applyLog(MatrixUtil.subtractFromScalar(1,A)));
				
				double sum = MatrixUtil.sumAll(MatrixUtil.add(f1,f2));
				
				double cost = (0-1.0/m) * sum;
				
				return cost;
	}
	
	/**
	 * The computation of backward prop to compute the derivatives.
	 * @param X
	 * @param Y
	 * @param activation
	 * @param cost
	 * @return
	 */
	private ForwardPropResult computeDerivatives(DoubleMatrix X, DoubleMatrix Y,
			DoubleMatrix activation, double cost) {
		
		int m = X.getRows();
		
		double derivB = (1.0/m) * MatrixUtil.sumAll(MatrixUtil.subtract(activation, Y));
		DoubleMatrix derivW = MatrixUtil.multiplyScalar(
					MatrixUtil.multiply(MatrixUtil.trans(MatrixUtil.subtract(activation, Y)),X), 1.0/m);
		
		ForwardPropResult result = new ForwardPropResult(derivB, derivW, cost);
		
		return result;
	}
	
	public ModelParams optimize(DoubleMatrix w, double b,
			DoubleMatrix X, DoubleMatrix Y, 
			int numIterations, double learningRate) {
		
		double cost=-1;
		
		for(int i=0;i<numIterations;i++) {
			ForwardPropResult result = forwardProp(w, b, X, Y);
			
			double db = result.getDerivBias();
			DoubleMatrix dw = result.getDerivWeights();
			
			cost = result.getCost();
			
			if(i % 50 == 0) {
				System.out.printf(".. Cost after iterations: %d is %f\n", i, cost);
				System.out.println("w.. ::" + w);
				System.out.println("b..::" + b);
			}
			w = MatrixUtil.subtract(w, MatrixUtil.multiplyScalar(dw, learningRate));
			b = b - learningRate * db;
		}
		
		return new ModelParams(w, b, cost);	
	}
	
	public void backwardProp() {
		
	}
	
	public void predict() {
		
	}
}
