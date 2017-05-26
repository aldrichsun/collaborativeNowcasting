/****************************
*
* Author: Yu Sun <yusun.aldrich@gmail.com>
* Date: 27 May 2017
* Version: 0.8
*
****************************/

A. This code re-implements the two-step collaborative nowcasting model introduced in the papers:
1. Yu Sun, Nicholas Jing Yuan, Xing Xie, Kieran McDonald, Rui Zhang. Collaborative Nowcasting for Contextual Recommendation, WWW, 1407-1418, 2016.
and
2. Yu Sun, Nicholas Jing Yuan, Xing Xie, Kieran McDonald, Rui Zhang. Collaborative Intent Prediction with Real-Time Contextual Data, ACM Transactions on Information Systems (TOIS), accepted, 2017.

B. The structure of the software is as follows.

	----folder dataExample: contains randomly generated example input data.
	
	----parafac2: an open-source version of parafac2 decomposition.
	
	----balancePanel.m: determines the balanced panel for estimating initial factors, loadings, and the transition matrix, system covariance, and measurement convariance.
	
	----evaluation.m: evaluates the model performance.
	
	----IndiviFactor.m: estimates the a posteriori factors.
	
	----main.m: the main software entry.
	
	----NowcastParafac2.m: the collaborative nowcasting model.
	
	----outliers_correction.m: adjusts for outliers and missing observations.
	
	----Parafac2Factors.m: obtains the collaborative factors.
	
	----preprocessPanel.m: preprocesses the input.
	
	----README.md: this file.
	
	----license.txt: the license for this software.

C. Please cite the related papers if your publications uses this open-source software.

D. If you find any bug or have any improvement suggestion, you are welcome to contact Yu Sun at yusun.aldrich@gmail.com.