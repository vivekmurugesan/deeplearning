package com.test.ml.commons.dtype;

import static org.junit.Assert.*;

import org.junit.Assert;
import org.junit.Test;

public class MatrixUtilTest {

	@Test
	public void testMultiplyIdentity() {
		double[][] a = {{1,2,3}, {4,5,6}, {7,8,9}};
		double[][] b = {{1,0,0}, {0,1,0}, {0,0,1}};
		
		DoubleMatrix one = new DoubleMatrix(3,3);
		DoubleMatrix two = new DoubleMatrix(3,3);
		
		one.setData(a);
		two.setData(b);
		
		DoubleMatrix three = MatrixUtil.multiply(one, two);
		
		Assert.assertTrue(one.equals(three));
	}
	
	@Test
	public void testMultiplyZero() {
		double[][] a = {{1,2,3}, {4,5,6}, {7,8,9}};
		double[][] b = {{0,0}, {0,0}, {0,0}};
		
		DoubleMatrix one = new DoubleMatrix(3,3);
		DoubleMatrix two = new DoubleMatrix(3,2);
		
		one.setData(a);
		two.setData(b);
		
		DoubleMatrix three = MatrixUtil.multiply(one, two);
		
		Assert.assertTrue(two.equals(three));
	}
	
	@Test
	public void testMultiply() {
		double[][] a = {{1,2,3}, {4,5,6}, {7,8,9}};
		double[][] b = {{1,1}, {1,1}, {1,1}};
		
		double[][] c = {{6,6}, {15,15}, {24,24}};
		
		DoubleMatrix one = new DoubleMatrix(3,3);
		DoubleMatrix two = new DoubleMatrix(3,2);
		
		DoubleMatrix expected = new DoubleMatrix(3,2);
		expected.setData(c);
		
		one.setData(a);
		two.setData(b);
		
		DoubleMatrix three = MatrixUtil.multiply(one, two);
		
		System.out.println("Expected.." + expected);
		System.out.println("Result.." + three);
		
		
		Assert.assertTrue(expected.equals(three));
	}
	
	@Test
	public void testAddZero() {
		double[][] a = {{1,2,3}, {4,5,6}, {7,8,9}};
		double x = 0;
		DoubleMatrix one = new DoubleMatrix(3,3);one.setData(a);
		DoubleMatrix expected = new DoubleMatrix(3,3);expected.setData(a);
		MatrixUtil.addScalar(one, x);
		
		Assert.assertTrue(expected.equals(one));
		
	}
	
	@Test
	public void testAddOne() {
		double[][] a = {{1,2,3}, {4,5,6}, {7,8,9}};
		double x = 1;
		double[][] b = {{2,3,4}, {5,6,7}, {8,9,10}};
		DoubleMatrix one = new DoubleMatrix(3,3);one.setData(a);
		DoubleMatrix expected = new DoubleMatrix(3,3);expected.setData(b);
		
		MatrixUtil.addScalar(one, x);
		
		Assert.assertTrue(expected.equals(one));
		
	}
	
	@Test
	public void testAddTen() {
		double[][] a = {{1,2,3}, {4,5,6}, {7,8,9}};
		double x = 10;
		double[][] b = {{11,12,13}, {14,15,16}, {17,18,19}};
		DoubleMatrix one = new DoubleMatrix(3,3);one.setData(a);
		DoubleMatrix expected = new DoubleMatrix(3,3);expected.setData(b);
		
		MatrixUtil.addScalar(one, x);
		
		Assert.assertTrue(expected.equals(one));
		
	}
	
	@Test
	public void testTrans() {
		double[][] a = {{1,2,3}, {4,5,6}, {7,8,9}};
		double[][] b = {{1,4,7}, {2,5,8}, {3,6,9}};
		
		DoubleMatrix one = new DoubleMatrix(3,3);
		DoubleMatrix two = new DoubleMatrix(3,3);
		
		one.setData(a);
		two.setData(b);
		
		DoubleMatrix result = MatrixUtil.trans(one);
		
		System.out.println("Input:" + one);
		System.out.println("Trans:" + result);
		
		Assert.assertTrue(result.equals(two));
	}
	
	@Test
	public void testTrans1() {
		double[][] a = {{1,2,3}, {4,5,6}};
		double[][] b = {{1,4}, {2,5}, {3,6}};
		
		DoubleMatrix one = new DoubleMatrix(2,3);
		DoubleMatrix two = new DoubleMatrix(3,2);
		
		one.setData(a);
		two.setData(b);
		
		DoubleMatrix result = MatrixUtil.trans(one);
		
		System.out.println("Input:" + one);
		System.out.println("Trans:" + result);
		
		Assert.assertTrue(result.equals(two));
	}
	
	@Test
	public void testElemWiseMultiplyIdentity() {
		double[][] a = {{1,2,3}, {4,5,6}, {7,8,9}};
		double[][] b = {{1,1,1}, {1,1,1}, {1,1,1}};
		
		DoubleMatrix one = new DoubleMatrix(3,3);
		DoubleMatrix two = new DoubleMatrix(3,3);
		
		one.setData(a);
		two.setData(b);
		
		DoubleMatrix three = MatrixUtil.elementWiseMultiply(one, two);
		
		System.out.println(".. Expected ::" + one);
		System.out.println(".. Result.." + three);
		
		Assert.assertTrue(one.equals(three));
	}
	
	@Test
	public void testElemWiseMultiplySquare() {
		double[][] a = {{1,2,3}, {4,5,6}, {7,8,9}};
		double[][] b = {{1,2,3}, {4,5,6}, {7,8,9}};
		
		double[][] expected = {{1,4,9}, {16,25,36}, {49,64,81}};
		
		DoubleMatrix one = new DoubleMatrix(3,3);
		DoubleMatrix two = new DoubleMatrix(3,3);
		DoubleMatrix expectedMat = new DoubleMatrix(3, 3);
		
		one.setData(a);
		two.setData(b);
		expectedMat.setData(expected);
		
		DoubleMatrix three = MatrixUtil.elementWiseMultiply(one, two);
		
		System.out.println(".. Expected ::" + one);
		System.out.println(".. Result.." + three);
		
		Assert.assertTrue(expectedMat.equals(three));
	}

	
	@Test
	public void testElemWiseMultiplyAdd1() {
		double[][] a = {{1,2,3}, {4,5,6}, {7,8,9}};
		double[][] b = {{1,2,3}, {4,5,6}, {7,8,9}};
		
		double[][] expected = {{2,4,6}, {8,10,12}, {14,16,18}};
		
		DoubleMatrix one = new DoubleMatrix(3,3);
		DoubleMatrix two = new DoubleMatrix(3,3);
		DoubleMatrix expectedMat = new DoubleMatrix(3, 3);
		
		one.setData(a);
		two.setData(b);
		expectedMat.setData(expected);
		
		DoubleMatrix three = MatrixUtil.add(one, two);
		
		System.out.println("Addition..");
		System.out.println(".. Expected ::" + expectedMat);
		System.out.println(".. Result.." + three);
		
		Assert.assertTrue(expectedMat.equals(three));
	}
}
