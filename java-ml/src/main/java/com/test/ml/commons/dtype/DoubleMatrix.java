package com.test.ml.commons.dtype;

import java.util.Arrays;
import java.util.Random;

/**
 * 
 * @author vivek
 *
 */
public class DoubleMatrix {

	private double[][] data;
	private int rows;
	private int cols;
	
	public DoubleMatrix(int rows, int cols) {
		this.rows = rows;
		this.cols = cols;
		data = new double[rows][cols];
	}
	
	public DoubleMatrix(double[][] data, int rows, int cols) {
		super();
		this.data = data;
		this.rows = rows;
		this.cols = cols;
	}
	
	public void initWithZero() {
		for(double[] row : data)
			for(int i=0;i<row.length;i++)
				row[i]=0.0;
	}
	
	public void initWithRandom(long seed) {
		Random rand = new Random(seed);
		for(double[] row : data)
			for(int i=0;i<row.length;i++)
				row[i] = rand.nextDouble()*0.01;
	}

	public double[][] getData() {
		return data;
	}

	public void setData(double[][] data) {
		this.data = data;
	}

	public int getRows() {
		return rows;
	}

	public void setRows(int rows) {
		this.rows = rows;
	}

	public int getCols() {
		return cols;
	}

	public void setCols(int cols) {
		this.cols = cols;
	}
	
	public double get(int row, int col) {
		return this.data[row][col];
	}
	
	public void set(int row, int col, double val) {
		this.data[row][col] = val;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + cols;
		result = prime * result + Arrays.deepHashCode(data);
		result = prime * result + rows;
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		DoubleMatrix other = (DoubleMatrix) obj;
		if (cols != other.cols)
			return false;
		if (!Arrays.deepEquals(data, other.data))
			return false;
		if (rows != other.rows)
			return false;
		return true;
	}
	
	public String toString() {
		StringBuilder bf = new StringBuilder();
		
		bf.append("{\n");
		for(double[] row : this.data)
			bf.append(Arrays.toString(row)).append("\n");
		bf.append("}");
		
		return bf.toString();
	}
	
	public void printDimension() {
		System.out.println(this.rows + "x" + this.cols);
	}
	
	public String getDimensionAsString() {
		return this.rows + "x" + this.cols;
	}
}
