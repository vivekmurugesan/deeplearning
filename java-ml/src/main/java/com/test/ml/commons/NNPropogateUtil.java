package com.test.ml.commons;

import java.io.PrintStream;
import java.util.Map;
import java.util.TreeMap;

import com.test.ml.commons.data.model.ActivationType;
import com.test.ml.commons.data.model.LayerCache;
import com.test.ml.commons.data.model.LayerGradient;
import com.test.ml.commons.data.model.nn.LayerParams;
import com.test.ml.commons.data.model.nn.NetworkLayout;
import com.test.ml.commons.dtype.DoubleMatrix;
import com.test.ml.commons.dtype.MatrixUtil;

/**
 * 
 * @author vivek
 *
 */
public class NNPropogateUtil extends PropogateUtil {

	/**
	 *  Sample operations..
	 *  Z1 = np.dot(W1, X) + b1   --> mxn ** nxh1 --> mxh1
    	A1 = np.tanh(Z1)		  --> mxh1
    	Z2 = np.dot(W2, A1) + b2  --> mxh1 ** h1x1 --> mx1
        A2 = sigmoid(Z2)		  --> mx1
     * For multiple layers.
     * for each i in layerCount:
     * 		Zi = MatrixUtil.multiply(Ai-1, Wi) + bi
     * 		Ai = Activation_i(Zi)
	 * @param X
	 * @param layerParams
	 */
	public DoubleMatrix forwardProp(DoubleMatrix X, DoubleMatrix Y, LayerParams[] layerParamsArr) {
		int m = X.getRows();
		DoubleMatrix A_prev = X;
		for(int i=0;i<layerParamsArr.length;i++) {
			LayerParams layerParams = layerParamsArr[i];
			DoubleMatrix Z = MatrixUtil.addRowVector(MatrixUtil.multiply(A_prev, MatrixUtil.trans(layerParams.getWeights())), layerParams.getBias());
			A_prev = MatrixUtil.applyActivation(Z, layerParams.getActivation());
			LayerCache lCache = new LayerCache(Z, A_prev, layerParams.getLayerNumber());
			layerParams.setLayerCache(lCache);
		}
		
		return A_prev;
	}
	
	/**
	 * The computation of backward prop to compute the derivatives.
	 * @param X
	 * @param Y
	 * @param activation
	 * @param cost
	 * @return
	 */
	private void backwardProp(DoubleMatrix X, DoubleMatrix Y,
			LayerParams[] layerParamsArr) {
		
		int m = X.getRows();
		int layerCount = layerParamsArr.length;
		
		LayerParams layerParams = layerParamsArr[layerCount-1];
		LayerCache lCache = layerParams.getLayerCache();

		DoubleMatrix aL = lCache.getA();
		
		/** dAL = - ((Y / AL) - ((1-Y) / (1-AL)) ) */
		
		DoubleMatrix f1 = MatrixUtil.elementWiseDivide(Y, aL);
		
		DoubleMatrix f2 = MatrixUtil.elementWiseDivide(MatrixUtil.subtractFromScalar(1, Y), 
						MatrixUtil.subtractFromScalar(1, aL));
		
		DoubleMatrix dAL = MatrixUtil.subtractFromScalar(0, 
				MatrixUtil.subtract(f1, f2));
		
		LayerGradient lGrad = new LayerGradient(layerCount-1);
		lGrad.setdA(dAL);
		
		layerParams.setLayerGrad(lGrad);
		/**
		 * grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")
    	 */
		
		
		DoubleMatrix aPrev = layerParamsArr[layerCount-2].getLayerCache().getA();
		dAL = linearActivationBackward(dAL, lCache.getZ(), lGrad, layerParams, aPrev);
		
		
		for(int lNumber=layerCount-2;lNumber>=0;lNumber--) {
			layerParams = layerParamsArr[lNumber];
			aPrev = (lNumber>=1)?layerParamsArr[lNumber-1].getLayerCache().getA() : X;
			lCache = layerParams.getLayerCache();
			// Getting the dA from the next layer gradient computed already in the backprop
			/* dAL = lGrad.getdA(); */   
			lGrad = new LayerGradient(lNumber);
			layerParams.setLayerGrad(lGrad);
			lGrad.setdA(dAL);
			dAL = linearActivationBackward(dAL, lCache.getZ(), lGrad, layerParams, aPrev);
			
		}
		
	}
	
	
	private DoubleMatrix linearActivationBackward(DoubleMatrix dA, DoubleMatrix Z, 
			LayerGradient lGrad, LayerParams layerParams, DoubleMatrix aPrev) {
		
		ActivationType actType = layerParams.getActivation().getType();
		DoubleMatrix dZ = null;
		switch(actType) {
		case sigmoid:
			dZ = MatrixUtil.applySigmoidBackward(Z, dA);
			break;
		case relu:
		case tanh:
		}
		lGrad.setdZ(dZ);
		return linearBackward(lGrad, layerParams, aPrev);
	}
	
	private DoubleMatrix linearBackward(LayerGradient lGrad, LayerParams layerParams, DoubleMatrix aPrev) {
		DoubleMatrix w = layerParams.getWeights();
		double[] b = layerParams.getBias();
		int m = aPrev.getRows();
		
		DoubleMatrix dZ = lGrad.getdZ();
		
		/**
		 * dZ --> mxn   aPrev --> mxm
		 */
		DoubleMatrix derivW = MatrixUtil.trans(MatrixUtil.multiplyScalar(MatrixUtil.multiply(MatrixUtil.trans(aPrev),dZ), 1.0/m));
		double[] derivB = MatrixUtil.meanAcrossCols(dZ);
		// FIXME --> Dimension mismatch 303x5 ** 1x5
		DoubleMatrix derivA = MatrixUtil.multiply(dZ, w);
		
		//lGrad.setdA(derivA); 
		lGrad.setdBias(derivB);
		lGrad.setdWeights(derivW);
		
		return derivA;
	}
	
	private void updateParameters(LayerParams[] layerParamsArr, double learningRate) {
		for(int i=0;i<layerParamsArr.length;i++) {
			LayerParams layerParams = layerParamsArr[i];
			LayerGradient lGrad = layerParams.getLayerGrad();
			DoubleMatrix dW = lGrad.getdWeights();
			layerParams.setWeights(MatrixUtil.subtract(layerParams.getWeights(),MatrixUtil.multiplyScalar(dW, learningRate)));
			
			double[] dB = lGrad.getdBias();
			double[] bias = layerParams.getBias();
			for(int j=0;j<dB.length;j++)
				dB[j] = bias[j] - dB[j]*learningRate;
			
		}
	}
	
	/**
	 * 1.initialize_parameters
     * 2.modelForward(X, parameters):
     * 3.computeCost(AL, Y):
     * 4.modelBackward(AL, Y, caches):
     * 5. update_parameters(parameters, grads, learning_rate):
     * @param X
	 * @param Y
	 * @param learningRate
	 * @param numIterations
	 * @param ps 
	 * @return
	 */
	public Map<Integer, Double> optimize(DoubleMatrix X, DoubleMatrix Y, NetworkLayout network,
			double learningRate, int numIterations, boolean printCost, PrintStream ps) {
		
		Map<Integer, Double> costs = new TreeMap<>();
		LayerParams[] layerParamsArr = network.getLayers();
		int m = X.getRows();
		
		for(int i=0;i<numIterations;i++) {
			
			
			DoubleMatrix AL = forwardProp(X, Y, layerParamsArr);
			double cost = computeCost(Y, AL, m);
			
			if(i == 0) {
				System.out.println("Layer Params Summary.."); network.printLayerParams();
			}
			
			if(printCost && i % 50 == 0) {
				System.out.println(".. Cost.. " + cost +"... on iteration::" + i);
				ps.println(i+","+cost);
				costs.put(i+1, cost);
			}
			
			backwardProp(X, Y, layerParamsArr);
			
			if(i == 0) {
			  System.out.println("Gradients summary.."); network.printGradients();
			}
			 
			updateParameters(layerParamsArr, learningRate);
			
		}
		
		return costs;
	}
	
	
}
