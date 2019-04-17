package com.test.ml.commons.data.model.nn;

import com.test.ml.commons.data.model.Activation;

/**
 * 
 * @author vivek
 *
 */
public class NetworkLayout {

	private int[] layerNodeCount;
	private int layerCount;
	private Activation[] activations;
	private int classCount;
	private int featureCount;
	
	private long randSeed;
	
	private LayerParams[] layers;
	
	public NetworkLayout(int layerCount, int[] layerNodeCount, int classCount, int featureCount,
			long randSeed, Activation[] activations) {
		this.layerCount = layerCount;
		this.layerNodeCount = layerNodeCount;
		this.classCount = classCount;
		this.featureCount = featureCount;
		this.randSeed = randSeed;
		this.activations = activations;
	}
	
	public void init() {
		this.layers = new LayerParams[layerCount];
		for(int i=0;i<layerCount;i++) {
			boolean outputLayer = (i==layerCount-1);
			layers[i] = new LayerParams(i, layerNodeCount[i],outputLayer, randSeed, activations[i]);
			int incomingEdges = (i>0)?layerNodeCount[i-1]:featureCount;
			layers[i].setIncomingEdges(incomingEdges);
			layers[i].initWeights();
		}
	}

	public int[] getLayerNodeCount() {
		return layerNodeCount;
	}

	public void setLayerNodeCount(int[] layerNodeCount) {
		this.layerNodeCount = layerNodeCount;
	}

	public int getLayerCount() {
		return layerCount;
	}

	public void setLayerCount(int layerCount) {
		this.layerCount = layerCount;
	}

	public Activation[] getActivations() {
		return activations;
	}

	public void setActivations(Activation[] activations) {
		this.activations = activations;
	}

	public int getClassCount() {
		return classCount;
	}

	public void setClassCount(int classCount) {
		this.classCount = classCount;
	}

	public int getFeatureCount() {
		return featureCount;
	}

	public void setFeatureCount(int featureCount) {
		this.featureCount = featureCount;
	}

	public long getRandSeed() {
		return randSeed;
	}

	public void setRandSeed(long randSeed) {
		this.randSeed = randSeed;
	}

	public LayerParams[] getLayers() {
		return layers;
	}

	public void setLayers(LayerParams[] layers) {
		this.layers = layers;
	}
	
	public void printLayerParams() {
		int layerCount = this.layers.length;
		
		System.out.println("_________________________________________");
		System.out.printf("| Layers\t|");
		for(int i=0;i<layerCount;i++)
			System.out.printf("%d\t|",i);
		
		System.out.println("\n_________________________________________");
		System.out.printf("| Weights\t|");
		for(int i=0;i<layerCount;i++)
			System.out.printf("%s\t|",this.layers[i].getWeights().getDimensionAsString());
		
		System.out.println("\n_________________________________________");
		System.out.printf("| Bias\t\t|");
		for(int i=0;i<layerCount;i++)
			System.out.printf("%d\t|",this.layers[i].getBias().length);
		
		System.out.println("\n_________________________________________");
		System.out.printf("| Z\t\t|");
		for(int i=0;i<layerCount;i++)
			System.out.printf("%s\t|",this.layers[i].getLayerCache().getZ().getDimensionAsString());
		
		System.out.println("\n_________________________________________");
		System.out.printf("| A\t\t|");
		for(int i=0;i<layerCount;i++)
			System.out.printf("%s\t|",this.layers[i].getLayerCache().getA().getDimensionAsString());
		System.out.println("\n_________________________________________");
	}
	
	public void printGradients() {
		int layerCount = this.layers.length;
		
		System.out.println("\n_________________________________________");
		System.out.printf("| Layers\t|");
		for(int i=0;i<layerCount;i++)
			System.out.printf("%d\t|",i);
		
		System.out.println("\n_________________________________________");
		System.out.printf("| dWeights\t|");
		for(int i=0;i<layerCount;i++)
			System.out.printf("%s\t|",this.layers[i].getLayerGrad().getdWeights().getDimensionAsString());
		
		System.out.println("\n_________________________________________");
		System.out.printf("| dBias\t\t|");
		for(int i=0;i<layerCount;i++)
			System.out.printf("%d\t|",this.layers[i].getLayerGrad().getdBias().length);
		
		System.out.println("\n_________________________________________");
		System.out.printf("| dZ\t\t|");
		for(int i=0;i<layerCount;i++)
			System.out.printf("%s\t|",this.layers[i].getLayerGrad().getdZ().getDimensionAsString());
		
		System.out.println("\n_________________________________________");
		System.out.printf("| dA\t\t|");
		for(int i=0;i<layerCount;i++)
			System.out.printf("%s\t|",this.layers[i].getLayerGrad().getdA().getDimensionAsString());
		System.out.println("\n_________________________________________");
		
	}
	
	
}
