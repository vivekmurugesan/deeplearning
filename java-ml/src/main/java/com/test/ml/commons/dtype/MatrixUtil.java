package com.test.ml.commons.dtype;

import java.util.Arrays;
import java.util.function.Function;

import com.test.ml.commons.MathUtil;
import com.test.ml.commons.data.model.Activation;

/**
 * 
 * @author vivek
 *
 */
public class MatrixUtil {
	
	public static DoubleMatrix multiply(DoubleMatrix a, DoubleMatrix b) {
		if(a.getCols() != b.getRows())
			throw new IllegalArgumentException("Dimension mismatch::" + 
					a.getRows()+"x"+a.getCols() +"**" + b.getRows() + "x" +b.getCols() );
		
		double[][] result = new double[a.getRows()][b.getCols()];
		double[][] aData = a.getData();
		double[][] bData = b.getData();
		
		for(int i=0;i<a.getRows();i++) {
			for(int j=0;j<b.getCols();j++) {
				result[i][j] = 0.0;
				for(int k=0;k<b.getRows();k++) {
					result[i][j] += aData[i][k] * bData[k][j];
				}
			}
		}
		
		DoubleMatrix resultMat = new DoubleMatrix(result, a.getRows(), b.getCols());
		
		return resultMat;
	}
	
	public static DoubleMatrix elementWiseMultiply(DoubleMatrix a, DoubleMatrix b) {
		if(a.getRows() != b.getRows() || 
				a.getCols() != b.getCols())
			throw new IllegalArgumentException("Dimension mismatch" + 
					a.getRows()+"x"+a.getCols() +"**" + b.getRows() + "x" +b.getCols() );
		
		double[][] result = new double[a.getRows()][a.getCols()];
		
		double[][] aData = a.getData();
		double[][] bData = b.getData();
		
		for(int i=0;i<a.getRows();i++)
			for(int j=0;j<a.getCols();j++) {
				if(aData[i][j] == 0.0 || bData[i][j] == 0.0)	// Don't care field.
					result[i][j] = 0.0;		
				else
					result[i][j] = aData[i][j] * bData[i][j];
			}
		
		DoubleMatrix resultMat = new DoubleMatrix(result, a.getRows(), b.getCols());
		
		return resultMat;
	}
	
	public static DoubleMatrix add(DoubleMatrix a, DoubleMatrix b) {
		if(a.getRows() != b.getRows() || 
				a.getCols() != b.getCols())
			throw new IllegalArgumentException("Dimension mismatch" + 
					a.getRows()+"x"+a.getCols() +"**" + b.getRows() + "x" +b.getCols() );
		
		double[][] result = new double[a.getRows()][a.getCols()];
		
		double[][] aData = a.getData();
		double[][] bData = b.getData();
		
		for(int i=0;i<a.getRows();i++)
			for(int j=0;j<a.getCols();j++)
				result[i][j] = aData[i][j] + bData[i][j];
		
		DoubleMatrix resultMat = new DoubleMatrix(a.getRows(), b.getCols());
		resultMat.setData(result);
		
		return resultMat;
	}
	
	public static DoubleMatrix subtract(DoubleMatrix a, DoubleMatrix b) {
		if(a.getRows() != b.getRows() || 
				a.getCols() != b.getCols())
			throw new IllegalArgumentException("Dimension mismatch" + 
					a.getRows()+"x"+a.getCols() +"--" + b.getRows() + "x" +b.getCols() );
		
		double[][] result = new double[a.getRows()][a.getCols()];
		
		double[][] aData = a.getData();
		double[][] bData = b.getData();
		
		for(int i=0;i<a.getRows();i++)
			for(int j=0;j<a.getCols();j++)
				result[i][j] = aData[i][j] - bData[i][j];
		
		DoubleMatrix resultMat = new DoubleMatrix(a.getRows(), b.getCols());
		resultMat.setData(result);
		
		return resultMat;
	}
	
	public static DoubleMatrix applyLog(DoubleMatrix a) {
		double[][] data = a.getData();
		double[][] result = new double[a.getRows()][a.getCols()];
		for(int j=0;j<a.getRows();j++) 
			for(int i=0;i<data[j].length;i++)
				result[j][i] = Math.log(data[j][i]);
		DoubleMatrix resultMat = new DoubleMatrix(result, a.getRows(), a.getCols());
		
		return resultMat;
	}
	
	public static DoubleMatrix applyAbs(DoubleMatrix a) {
		double[][] data = a.getData();
		double[][] result = new double[a.getRows()][a.getCols()];
		for(int j=0;j<a.getRows();j++) 
			for(int i=0;i<data[j].length;i++)
				result[j][i] = Math.abs(data[j][i]);
		DoubleMatrix resultMat = new DoubleMatrix(result, a.getRows(), a.getCols());
		
		return resultMat;
	}
	
	public static DoubleMatrix applyActivation(DoubleMatrix a, Activation activation) {
		double[][] data = a.getData();
		double[][] result = new double[a.getRows()][a.getCols()];
		for(int j=0;j<a.getRows();j++) 
			for(int i=0;i<data[j].length;i++)
				result[j][i] = activation.apply(data[j][i]);
		DoubleMatrix resultMat = new DoubleMatrix(result, a.getRows(), a.getCols());
		
		return resultMat;
	}
	
	public static DoubleMatrix applySigmoid(DoubleMatrix a) {
		double[][] data = a.getData();
		double[][] result = new double[a.getRows()][a.getCols()];
		for(int j=0;j<a.getRows();j++) 
			for(int i=0;i<data[j].length;i++)
				result[j][i] = MathUtil.sigmoid(data[j][i]);
		DoubleMatrix resultMat = new DoubleMatrix(result, a.getRows(), a.getCols());
		
		return resultMat;
	}
	
	public static DoubleMatrix subtractScalar(DoubleMatrix a, double x) {
		double[][] data = a.getData();
		double[][] result = new double[a.getRows()][a.getCols()];
		for(int j=0;j<a.getRows();j++) 
			for(int i=0;i<data[j].length;i++)
				result[j][i] = data[j][i] - x;
		DoubleMatrix resultMat = new DoubleMatrix(result, a.getRows(), a.getCols());
		
		return resultMat;
	}
	
	public static DoubleMatrix subtractFromScalar(double x, DoubleMatrix a) {
		double[][] data = a.getData();
		double[][] result = new double[a.getRows()][a.getCols()];
		for(int j=0;j<a.getRows();j++) 
			for(int i=0;i<data[j].length;i++)
				result[j][i] = x - data[j][i];
		DoubleMatrix resultMat = new DoubleMatrix(result, a.getRows(), a.getCols());
		
		return resultMat;
	}
	
	public static DoubleMatrix addScalar(DoubleMatrix a, double x) {
		double[][] data = a.getData();
		double[][] result = new double[a.getRows()][a.getCols()];
		for(int j=0;j<a.getRows();j++) 
			for(int i=0;i<data[j].length;i++)
				result[j][i] = data[j][i] + x;
		DoubleMatrix resultMat = new DoubleMatrix(result, a.getRows(), a.getCols());
		
		return resultMat;
	}
	
	public static DoubleMatrix multiplyScalar(DoubleMatrix a, double x) {
		double[][] data = a.getData();
		double[][] result = new double[a.getRows()][a.getCols()];
		for(int j=0;j<a.getRows();j++) 
			for(int i=0;i<data[j].length;i++)
				result[j][i] = data[j][i] * x;
		DoubleMatrix resultMat = new DoubleMatrix(result, a.getRows(), a.getCols());
		
		return resultMat;
	}
	
	public static double sumAll(DoubleMatrix a) {
		double result = 0.0;
		double[][] data = a.getData();
		for(double[] row : data) 
			for(int i=0;i<row.length;i++)
				if(row[i] != Double.NaN)
					result += row[i];
		
		return result;
	}
	
	public static DoubleMatrix trans(DoubleMatrix a) {
		double[][] data = a.getData();
		double[][] resData = new double[a.getCols()][a.getRows()];
		
		for(int i=0;i<a.getCols();i++) 
			for(int j=0;j<a.getRows();j++) 
				resData[i][j] = data[j][i];
		
		DoubleMatrix result = new DoubleMatrix(resData,a.getCols(), a.getRows());
		
		return result;
	}
	
	public static DoubleMatrix addRowVector(DoubleMatrix a, double[] rowVector) {
		double[][] data = a.getData();
		double[][] result = new double[a.getRows()][a.getCols()];
		
		for(int i=0;i<data.length;i++) 
			for(int j=0;j<data[i].length;j++)
				result[i][j] = data[i][j]+rowVector[j];
		
		DoubleMatrix resultMat = new DoubleMatrix(result, a.getRows(), a.getCols());
		
		return resultMat;
	}
	
	public static DoubleMatrix applySigmoidBackward(DoubleMatrix Z, DoubleMatrix dA) {
		double[][] sData = new double[Z.getRows()][Z.getCols()];
		double[][] data = Z.getData();
		double[][] dAData = dA.getData();
		
		double[][] dZData = new double[Z.getRows()][Z.getCols()];
		
		for(int i=0;i<sData.length;i++) {
			for(int j=0;j<sData[i].length;j++) {
				sData[i][j] = 1.0/(1.0+Math.exp(0-data[i][j]));
				dZData[i][j] = dAData[i][j] * sData[i][j] * (1-sData[i][j]);
			}
		}
		
		DoubleMatrix dZ = new DoubleMatrix(dZData,Z.getRows(), Z.getCols());
		
		return dZ;
	}
	
	public static double[] sumAcrossCols(DoubleMatrix a) {
		double[][] data = a.getData();
		double[] result = new double[a.getCols()];
		Arrays.fill(result, 0.0);
		
		for(int i=0;i<result.length;i++)
			for(int j=0;j<data.length;j++) // iterate through the rows of a given col
				result[i] += data[j][i];
		
		return result;
	}
	
	public static double[] meanAcrossCols(DoubleMatrix a) {
		double[][] data = a.getData();
		double[] result = new double[a.getCols()];
		Arrays.fill(result, 0.0);
		int rows = a.getRows();
		
		for(int i=0;i<result.length;i++)
			for(int j=0;j<rows;j++) // iterate through the rows of a given col
				result[i] += data[j][i];
		
		for(int i=0;i<result.length;i++)
			result[i] /= rows;
		
		return result;
	}
	
	public static DoubleMatrix elementWiseDivide(DoubleMatrix a, DoubleMatrix b) {
		double[][] aData = a.getData();
		double[][] bData = b.getData();
		double[][] result = new double[a.getRows()][a.getCols()];
		int rows = a.getRows();
		int cols = a.getCols();
		
		for(int i=0;i<rows;i++)
			for(int j=0;j<cols;j++)
				result[i][j] = aData[i][j]/bData[i][j];
		
		DoubleMatrix resultMat = new DoubleMatrix(result,rows, cols);
		
		return resultMat;
	}
}
