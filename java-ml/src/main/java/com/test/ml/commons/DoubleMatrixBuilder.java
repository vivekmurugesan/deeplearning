package com.test.ml.commons;

import java.util.ArrayList;
import java.util.List;

import com.test.ml.commons.dtype.DoubleMatrix;

/**
 * 
 * @author vivek
 *
 */
public class DoubleMatrixBuilder {

	private List<double[]> columnList;
	private int rows;
	
	public DoubleMatrixBuilder(int rows) {
		this.columnList = new ArrayList<>();
		this.rows = rows;
	}
	
	public void addColData(double[] colData) {
		if(colData.length < rows)
			throw new IllegalArgumentException(
					".. Insufficient number of elements:expected::" +  rows
					+"--> passed::" + colData.length); 
		this.columnList.add(colData);
	}
	
	public DoubleMatrix build() {
		int cols = columnList.size();
		DoubleMatrix result = new DoubleMatrix(rows, cols);
		int cIndex = 0;
		for(double[] colData : columnList) {
			for(int rIndex=0;rIndex<rows;rIndex++)
				result.set(rIndex,cIndex, colData[rIndex]);
			cIndex++;
		}
		
		return result;
	}
	
	
}
