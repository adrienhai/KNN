#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>
#include "armadillo.hpp"

struct pairKNN {
  double dist;
  int label;
};


bool sortDist(pairKNN a, pairKNN b){
  return a.dist < b.dist;
}

int main(int argc, char const *argv[]) {
  // pairKNN a;
  // a.dist=3.4;
  // a.label=1;
  // pairKNN b;
  // b.dist=4.6;
  // b.label=-1;
  // std::cout << sortDist(a,b) << '\n';

  arma::vec dataY ;
	dataY.load("dataY.dat") ;
  arma::mat dataX;
	dataX.load("dataX.dat");
	arma::mat dataXtest;
	dataXtest.load("dataXtest.dat") ;

  int k = 5;
	int i = 0 ;

	arma::vec XtestLabel = arma::vec(dataXtest.n_rows) ;

	i = 0;
	while(i < dataXtest.n_rows ){
		std::vector<pairKNN> pair_vector(dataX.n_rows) ;

		int j = 0;
		while(j < dataX.n_rows)	{
			pairKNN P ;
			P.dist = norm(dataX.row(j) - dataXtest.row(i));
			P.label = dataY(j);
			pair_vector[j] = P;
			j++;
		}

		std::sort(pair_vector.begin(),pair_vector.end(), sortDist);

		int minus_ones = 0;
		int ones = 0;

		for(int l = 0; l<k; l++)
		{
			if(pair_vector[l].label == 1)
			{
				ones++ ;
			}
			else
			{
				minus_ones ++;
			}
		}


		if(ones > minus_ones)
		{
			XtestLabel(i) = 1;
		}
		else
		{
			XtestLabel(i) = -1;
		}
		std::cout << XtestLabel(i) << "\n" ;

		i++;
	}

	i=0;

	std::ofstream write_output("NN.dat") ;
	assert(write_output.is_open());
	for(i = 0; i<dataXtest.n_rows; i++)
	{
		write_output << XtestLabel(i) << "\n" ;
	}
	write_output.close();


  return 0;
}
