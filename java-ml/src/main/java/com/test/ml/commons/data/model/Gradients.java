package com.test.ml.commons.data.model;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.test.ml.commons.dtype.DoubleMatrix;

/**
 * 
 * @author vivek
 *
 */
public class Gradients {

	private Map<Integer, LayerGradient> layerGrads;
	
	public Gradients() {
		layerGrads = new HashMap<>();
	}

	public Map<Integer, LayerGradient> getLayerGrads() {
		return layerGrads;
	}

	public void setLayerGrads(Map<Integer, LayerGradient> layerGrads) {
		this.layerGrads = layerGrads;
	}
	
}
