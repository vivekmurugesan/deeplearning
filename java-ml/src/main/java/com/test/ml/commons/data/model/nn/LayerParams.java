package com.test.ml.commons.data.model.nn;

import java.util.Arrays;

import com.test.ml.commons.data.model.Activation;
import com.test.ml.commons.data.model.LayerCache;
import com.test.ml.commons.data.model.LayerGradient;
import com.test.ml.commons.dtype.DoubleMatrix;

/**
 * 
 * @author vivek
 *
 */
public class LayerParams {

	private int layerNumber;
	private int nodeCount;
	private boolean outputLayer;
	private int incomingEdges;
	
	private Activation activation;
	
	private long randSeed;
	
	private DoubleMatrix weights;
	private double[] bias;
	
	private transient LayerCache layerCache;
	private transient LayerGradient layerGrad;
	
	public LayerParams(int layerNumber, int nodeCount,
			boolean outputLayer, long randSeed, Activation activation) {
		this.layerNumber = layerNumber;
		this.nodeCount = nodeCount;
		this.outputLayer = outputLayer;
		this.randSeed = randSeed;
		this.activation = activation;
		this.layerCache = new LayerCache(layerNumber);
	}
	
	public void setIncomingEdges(int incomingEdges) {
		this.incomingEdges = incomingEdges;
	}

	public void initWeights() {
		this.weights = new DoubleMatrix(this.nodeCount, this.incomingEdges);
		this.weights.initWithRandom(randSeed);
		this.bias = new double[this.nodeCount];
		Arrays.fill(this.bias, 0.0);
	}

	public int getLayerNumber() {
		return layerNumber;
	}

	public void setLayerNumber(int layerNumber) {
		this.layerNumber = layerNumber;
	}

	public int getNodeCount() {
		return nodeCount;
	}

	public void setNodeCount(int nodeCount) {
		this.nodeCount = nodeCount;
	}

	public boolean isOutputLayer() {
		return outputLayer;
	}

	public void setOutputLayer(boolean outputLayer) {
		this.outputLayer = outputLayer;
	}

	public long getRandSeed() {
		return randSeed;
	}

	public void setRandSeed(long randSeed) {
		this.randSeed = randSeed;
	}

	public DoubleMatrix getWeights() {
		return weights;
	}

	public void setWeights(DoubleMatrix weights) {
		this.weights = weights;
	}

	public double[] getBias() {
		return bias;
	}

	public void setBias(double[] bias) {
		this.bias = bias;
	}

	public int getIncomingEdges() {
		return incomingEdges;
	}

	public Activation getActivation() {
		return activation;
	}

	public void setActivation(Activation activation) {
		this.activation = activation;
	}

	public LayerCache getLayerCache() {
		return layerCache;
	}

	public void setLayerCache(LayerCache layerCache) {
		this.layerCache = layerCache;
	}

	public LayerGradient getLayerGrad() {
		return layerGrad;
	}

	public void setLayerGrad(LayerGradient layerGrad) {
		this.layerGrad = layerGrad;
	}
	
	
	
}
