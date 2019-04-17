package com.test.ml.commons;

/**
 * 
 * @author vivek
 *
 */
public class MathUtil {

	public static double sigmoid(double z) {
		double s = 1/(1+Math.exp(0-z));
		
		return s;
	}
	
	public static void main(String[] args) {
		
	}

}
