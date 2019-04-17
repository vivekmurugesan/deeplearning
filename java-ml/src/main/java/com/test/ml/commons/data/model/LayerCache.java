package com.test.ml.commons.data.model;

import com.test.ml.commons.dtype.DoubleMatrix;

/**
 * 
 * @author vivek
 *
 */
public class LayerCache {

	private DoubleMatrix z;
	private DoubleMatrix a;
	
	private int layerNumber;

	public LayerCache(int layerNumber) {
		this.layerNumber = layerNumber;
	}
	
	public LayerCache(DoubleMatrix z, DoubleMatrix a, int layerNumber) {
		super();
		this.z = z;
		this.a = a;
		this.layerNumber = layerNumber;
	}



	public DoubleMatrix getZ() {
		return z;
	}

	public void setZ(DoubleMatrix z) {
		this.z = z;
	}

	public DoubleMatrix getA() {
		return a;
	}

	public void setA(DoubleMatrix a) {
		this.a = a;
	}

	public int getLayerNumber() {
		return layerNumber;
	}

	public void setLayerNumber(int layerNumber) {
		this.layerNumber = layerNumber;
	}
	
	
	
}
