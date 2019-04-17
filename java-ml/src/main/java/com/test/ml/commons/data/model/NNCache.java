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
public class NNCache {

	private Map<Integer, LayerCache> layerCaches;
	
	public NNCache() {
		layerCaches = new HashMap<>();
	}

	public Map<Integer, LayerCache> getLayerCaches() {
		return layerCaches;
	}

	public void setLayerCaches(Map<Integer, LayerCache> layerCaches) {
		this.layerCaches = layerCaches;
	}
	
	
}
