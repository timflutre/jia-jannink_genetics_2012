#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>

int i, j, k, l;
int rmvnorm(const gsl_rng *r, const int n, const gsl_vector *mean, const gsl_matrix *var, gsl_vector *result) {
	/* multivariate normal distribution random number generator */
	/*
	 *    n       dimension of the random vetor
	 *    mean    vector of means of size n
	 *    var     variance matrix of dimension n x n
	 *    result  output variable with a sigle random vector normal distribution generation
	 */
	gsl_matrix *work = gsl_matrix_calloc(n, n);
	gsl_matrix_memcpy(work, var);
	gsl_linalg_cholesky_decomp(work);
	for (k = 0; k < n; k++)
		gsl_vector_set(result, k, gsl_ran_ugaussian(r));

	gsl_blas_dtrmv(CblasLower, CblasNoTrans, CblasNonUnit, work, result);
	gsl_vector_add(result, mean);
	gsl_matrix_free(work);
	return 0;
}

int rwishart(const gsl_rng *r, const int n, const int dof, const gsl_matrix *scale, gsl_matrix *result) {
	/* Wishart distribution random number generator */
	/*
	 *    n        gives the dimension of the random matrix
	 *    dof      degrees of freedom
	 *    scale    scale matrix of dimension n x n
	 *    result   output variable with a single random matrix Wishart distribution generation
	 */
	gsl_matrix *work = gsl_matrix_calloc(n, n);

	for (k = 0; k < n; k++) {
		gsl_matrix_set(work, k, k, sqrt(gsl_ran_chisq(r, (dof - k))));
		for (l = 0; l < k; l++) {
			gsl_matrix_set(work, k, l, gsl_ran_ugaussian(r));
		}
	}
	gsl_matrix_memcpy(result, scale);
	gsl_linalg_cholesky_decomp(result);
	gsl_blas_dtrmm(CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, 1.0, result, work);
	gsl_blas_dsyrk(CblasUpper, CblasNoTrans, 1.0, work, 0.0, result);
	for (i = 0; i < n; i++) {
		for (j = i + 1; j < n; j++) {
			gsl_matrix_set(result, j, i, gsl_matrix_get(result, i, j));
		}
	}
	gsl_matrix_free(work);
	return 0;
}

void mat_transpose(gsl_matrix *a, gsl_matrix *b) {
	for (i = 0; i < a->size1; i++) {
		for (j = 0; j < a->size2; j++) {
			gsl_matrix_set(b, j, i, gsl_matrix_get(a, i, j));
		}
	}
}


void mm_mul(gsl_matrix *a, gsl_matrix *b, gsl_matrix *c)/* Function of matrix multiply by another matrix */
{
	double sum ;
	for (i = 0; i < a->size1; i++) {
		for (j = 0; j < b->size2; j++) {
			sum = 0.0;
			for (k = 0; k < a->size2; k++) {
				sum += gsl_matrix_get(a, i, k) * gsl_matrix_get(b, k, j);
			}
			gsl_matrix_set(c, i, j, sum);
		}
	}
}/* end of mm_mul function */

double get_det(gsl_matrix * a) {
	int sign = 0;
	double det = 0.0;
	int row_sq = a->size1;
	gsl_permutation * p = gsl_permutation_calloc(row_sq);
	gsl_matrix * tmp_ptr = gsl_matrix_calloc(row_sq, row_sq);
	int * signum = &sign;
	gsl_matrix_memcpy(tmp_ptr, a);
	gsl_linalg_LU_decomp(tmp_ptr, p, signum);
	det = gsl_linalg_LU_det(tmp_ptr, *signum);
	gsl_permutation_free(p);
	gsl_matrix_free(tmp_ptr);
	return det;
}

void mat_inverse(gsl_matrix * a, gsl_matrix * inverse) {
	int sign = 0;
	int row_sq = a->size1;
	gsl_permutation * p = gsl_permutation_calloc(row_sq);
	gsl_matrix * tmp_ptr = gsl_matrix_calloc(row_sq, row_sq);
	int * signum = &sign;
	gsl_matrix_memcpy(tmp_ptr, a);
	gsl_linalg_LU_decomp(tmp_ptr, p, signum);
	gsl_linalg_LU_invert(tmp_ptr, p, inverse);
	gsl_permutation_free(p);
	gsl_matrix_free(tmp_ptr);

}

void cal_scalec_post(gsl_matrix * scalec_ptr, gsl_matrix * y_ptr, gsl_matrix * scalec_post_ptr) {
	gsl_matrix * tmp_ptr = gsl_matrix_calloc(y_ptr->size2, y_ptr->size1);
	mat_transpose(y_ptr, tmp_ptr);
	mm_mul(tmp_ptr, y_ptr, scalec_post_ptr);
	gsl_matrix_add(scalec_post_ptr, scalec_ptr);
	gsl_matrix_free(tmp_ptr);
}

void adjusty(gsl_matrix *xm_ptr, gsl_matrix *bm_ptr, int c, gsl_matrix *y_ptr) {
	gsl_matrix * tmp_ptr = gsl_matrix_calloc(y_ptr->size1, y_ptr->size2);
	mm_mul(xm_ptr, bm_ptr, tmp_ptr);
	gsl_matrix_scale(tmp_ptr, c);
	gsl_matrix_add(y_ptr, tmp_ptr);
	gsl_matrix_free(tmp_ptr);
}

//Multivariate BayesCPi implemented by MCMC sampling

void mtbc_mcmc(int *thin, int *numiter,int *ntraits, int *nind, int *nmarkers, gsl_matrix *y_ptr, gsl_matrix *x_ptr, gsl_matrix *b_ptr, gsl_matrix *bStore_ptr) {
	int iter, locus, nLoci ;
	nLoci = 0;

	double pi, logDelta0, logDelta1, probDelta1, u, aa, bb;
	pi= 0.5;

	//Initialize: scalec is the prior scale matrix; varEffects are the common posterior marker effect variance
	gsl_matrix * scalec_ptr = gsl_matrix_calloc(*ntraits, *ntraits);
	gsl_matrix * varEffects_ptr = gsl_matrix_calloc(*ntraits, *ntraits);
	gsl_matrix * tmp_square_ptr = gsl_matrix_calloc(*ntraits, *ntraits);

	for (i = 0; i < *ntraits; i++) {
		for (j = 0; j < *ntraits; j++) {
			if (i == j) {
				gsl_matrix_set(scalec_ptr, i, j, 0.1);
				gsl_matrix_set(varEffects_ptr, i, j, 0.1);
			} else {
				gsl_matrix_set(scalec_ptr, i, j, 0);
				gsl_matrix_set(varEffects_ptr, i, j, 0);
			}
		}
	}


	gsl_matrix * xm_ptr = gsl_matrix_calloc(*nind, 1);
	gsl_matrix * bm_ptr = gsl_matrix_calloc(1, *ntraits);

	gsl_matrix * scalec_post_ptr = gsl_matrix_calloc(*ntraits, *ntraits);
	gsl_matrix * vare_ptr = gsl_matrix_calloc(*ntraits, *ntraits);
	gsl_matrix * vare_mean_ptr = gsl_matrix_calloc(*ntraits, *ntraits);

	gsl_matrix * xm_t_ptr = gsl_matrix_calloc(1, *nind);
	gsl_matrix * rhs_ptr = gsl_matrix_calloc(1, *ntraits);
	gsl_matrix * xpx = gsl_matrix_calloc(1, 1);
	gsl_matrix * v0_ptr = gsl_matrix_calloc(*ntraits, *ntraits);
	gsl_matrix * rhs_t_ptr = gsl_matrix_calloc(*ntraits, 1);
	gsl_matrix * tmp1_ptr = gsl_matrix_calloc(*ntraits, *ntraits);
	gsl_matrix * tmp2_ptr = gsl_matrix_calloc(1, *ntraits);
	gsl_matrix * tmp4_ptr = gsl_matrix_calloc(*ntraits, *ntraits);
	gsl_matrix * mean_matrix_ptr = gsl_matrix_calloc(*ntraits, 1);
	gsl_vector * mean_vector_ptr = gsl_vector_calloc(*ntraits);
	gsl_matrix * vare_inv_ptr = gsl_matrix_calloc(*ntraits, *ntraits);
	gsl_matrix * varEffects_inv_ptr = gsl_matrix_calloc(*ntraits, *ntraits);
	gsl_matrix * lhs_ptr = gsl_matrix_calloc(*ntraits, *ntraits);
	gsl_matrix * invLhs_ptr = gsl_matrix_calloc(*ntraits, *ntraits);
	gsl_vector * randnum = gsl_vector_calloc(*ntraits);

	gsl_vector *rand_num = gsl_vector_calloc(*ntraits), *mean_num = gsl_vector_calloc(*ntraits);
	gsl_matrix *m = gsl_matrix_calloc(*ntraits, *ntraits), *rm = gsl_matrix_calloc(*ntraits, *ntraits);
	gsl_rng *r;
	r = gsl_rng_alloc(gsl_rng_mt19937);
	gsl_vector_set(rand_num, 0, 0.1);
	gsl_vector_set(rand_num, 1, 0.1);



	//The mcmc iterations start from here
	for (iter = 0; iter < *numiter; iter++) {
		gsl_rng_set(r, rand());
		nLoci = 0;
		//sample the vare from here
		cal_scalec_post(scalec_ptr, y_ptr, scalec_post_ptr);
		mat_inverse(scalec_post_ptr, m);
		//sample vare via rwish sampling
		rwishart(r, *ntraits, (*ntraits+1 + *nind), m, rm);
		for (i = 0; i < *ntraits; i++) {
			for (j = i + 1; j < *ntraits; j++) {
				gsl_matrix_set(rm, j, i, gsl_matrix_get(rm, i, j));
			}
		}

		mat_inverse(rm, vare_ptr);
		//end of vare estimation



		//sample the intercept from here
		for (i = 0; i < *nind; i++) {
			gsl_matrix_set(xm_ptr, i, 0, gsl_matrix_get(x_ptr, i, 0));
		}
		//printf("this is a test\n");
		for (i = 0; i < *ntraits; i++)
			gsl_matrix_set(bm_ptr, 0, i, gsl_matrix_get(b_ptr, 0, i));

		adjusty(xm_ptr, bm_ptr, 1, y_ptr);
		gsl_matrix_memcpy(vare_mean_ptr, vare_ptr);

		for (j = 0; j < *ntraits; j++) {
			gsl_vector_set(mean_num, j, 0);
			for (i = 0; i < *nind; i++) {
				gsl_vector_set(mean_num, j, gsl_vector_get(mean_num, j) + gsl_matrix_get(y_ptr, i,j));
			}
			gsl_vector_set(mean_num, j, gsl_vector_get(mean_num, j) / (*nind));
		}

		for (i = 0; i < *ntraits; i++) {
			for (j = 0; j < *ntraits; j++) {
				gsl_matrix_set(vare_mean_ptr, i, j, gsl_matrix_get(vare_ptr, i, j) / (*nind));
			}
		}
		gsl_rng_set(r, rand());
		rmvnorm(r, *ntraits, mean_num, vare_mean_ptr, rand_num);

		for (i = 0; i < *ntraits; i++) {
			gsl_matrix_set(b_ptr, 0, i, gsl_vector_get(rand_num, i));
			gsl_matrix_set(bm_ptr, 0, i, gsl_vector_get(rand_num, i));
		}

		adjusty(xm_ptr, bm_ptr, -1, y_ptr);
		//end of intercept estimation

		//Estimate marker effect from here
		gsl_matrix_memcpy(scalec_post_ptr, scalec_ptr);
		for (locus = 1; locus < (*nmarkers) + 1; locus++) {

			for (i = 0; i < *nind; i++)
				gsl_matrix_set(xm_ptr, i, 0, gsl_matrix_get(x_ptr, i, locus));
			for (i = 0; i < *ntraits; i++)
				gsl_matrix_set(bm_ptr, 0, i, gsl_matrix_get(b_ptr, locus, i));
			adjusty(xm_ptr, bm_ptr, 1, y_ptr);

			mat_transpose(xm_ptr, xm_t_ptr);

			//printf("this is a test\n");
			mm_mul(xm_t_ptr, y_ptr, rhs_ptr);

			mm_mul(xm_t_ptr, xm_ptr, xpx);

			gsl_matrix_memcpy(v0_ptr, vare_ptr);
			gsl_matrix_scale(v0_ptr, gsl_matrix_get(xpx, 0, 0));

			mat_transpose(rhs_ptr, rhs_t_ptr);

			mat_inverse(v0_ptr, tmp1_ptr);

			mm_mul(rhs_ptr, tmp1_ptr, tmp2_ptr);
			mm_mul(tmp2_ptr, rhs_t_ptr, xpx);
			double det;
			det= get_det(v0_ptr);

			logDelta0 = -0.5 * (log(det) + gsl_matrix_get(xpx, 0, 0)) + log(pi);

			gsl_matrix_memcpy(tmp1_ptr, varEffects_ptr);
			mm_mul(xm_t_ptr, xm_ptr, xpx);
			gsl_matrix_scale(tmp1_ptr, pow(gsl_matrix_get(xpx, 0, 0), 2));
			gsl_matrix_add(tmp1_ptr, v0_ptr);
			mat_inverse(tmp1_ptr, tmp4_ptr);
			mm_mul(rhs_ptr, tmp4_ptr, tmp2_ptr);
			mm_mul(tmp2_ptr, rhs_t_ptr, xpx);
			det=get_det(tmp1_ptr);
			logDelta1 = -0.5 * (log(det) + gsl_matrix_get(xpx, 0, 0)) + log(1-(pi));


			probDelta1 = 1.0 / (1.0 + exp((logDelta0) - (logDelta1)));
			u = rand() / ((double) RAND_MAX + 1);

			if (u < probDelta1) {
				nLoci = nLoci + 1;

				mm_mul(xm_t_ptr, xm_ptr, xpx);
				mat_inverse(vare_ptr, lhs_ptr);
				mat_inverse(varEffects_ptr, varEffects_inv_ptr);
				gsl_matrix_scale(lhs_ptr, gsl_matrix_get(xpx, 0, 0));
				gsl_matrix_add(lhs_ptr, varEffects_inv_ptr);
				mat_inverse(lhs_ptr, invLhs_ptr);

				mat_inverse(vare_ptr, vare_inv_ptr);
				mm_mul(invLhs_ptr, vare_inv_ptr, varEffects_inv_ptr);
				mm_mul(varEffects_inv_ptr, rhs_t_ptr, mean_matrix_ptr);
				for (i = 0; i < varEffects_ptr->size1; i++)
					gsl_vector_set(mean_vector_ptr, i, gsl_matrix_get(mean_matrix_ptr, i, 0));
				gsl_vector_memcpy(randnum, mean_vector_ptr);

				gsl_rng_set(r, rand());
				rmvnorm(r, varEffects_ptr->size1, mean_vector_ptr, invLhs_ptr, randnum);
				for (i = 0; i < varEffects_ptr->size1; i++)
					gsl_matrix_set(bm_ptr, 0, i, gsl_vector_get(randnum, i));
			} else {
				for (i = 0; i < varEffects_ptr->size1; i++)
					gsl_matrix_set(bm_ptr, 0, i, 0);
			}

			for (i = 0; i < *ntraits; i++)
				gsl_matrix_set(b_ptr, locus, i, gsl_matrix_get(bm_ptr, 0, i));
			adjusty(xm_ptr, bm_ptr, -1, y_ptr);

			gsl_matrix * bm_t_ptr = gsl_matrix_calloc(bm_ptr->size2, bm_ptr->size1);
			mat_transpose(bm_ptr, bm_t_ptr);
			mm_mul(bm_t_ptr, bm_ptr, tmp_square_ptr);
			gsl_matrix_add(scalec_post_ptr, tmp_square_ptr);
			gsl_matrix_free(bm_t_ptr);

		}


		//printf("this is a test\n");
		//estimate common variance

		mat_inverse(scalec_post_ptr, tmp_square_ptr);
		gsl_rng_set(r, rand());
		rwishart(r, *ntraits, (*ntraits+1 + nLoci), tmp_square_ptr, rm);

		for (i = 0; i < *ntraits; i++) {
			for (j = i + 1; j < *ntraits; j++) {
				gsl_matrix_set(rm, j, i, gsl_matrix_get(rm, i, j));
			}
		}

		mat_inverse(rm, varEffects_ptr);
		//end of common variance estimation


		//sample Pi
		aa =  (double) (*nmarkers - nLoci + 1);
		bb =  (double) (nLoci + 1);
		pi = gsl_ran_beta(r, aa, bb);

		if((iter+1) % (*thin) ==0) {
			for (j = 0; j < *nmarkers + 1; j++) {
				for (k = 0; k < *ntraits; k++) {
					gsl_matrix_set(bStore_ptr, (iter+1)/(*thin)-1, j * (*ntraits) + k,
							gsl_matrix_get(b_ptr, j, k));
				}
			}
		}//end of bStore store for each iteration
		if((iter+1) % 100 == 0) {
							printf("iteration %d \n",iter);
						}
	} //end of the iteration

	gsl_matrix_free(scalec_ptr);
	gsl_matrix_free(varEffects_ptr);
	gsl_matrix_free(tmp_square_ptr);
	gsl_matrix_free(xm_ptr);
	gsl_matrix_free(bm_ptr);
	gsl_matrix_free(scalec_post_ptr);
	gsl_matrix_free(vare_ptr);
	gsl_matrix_free(vare_mean_ptr);
	gsl_matrix_free(xm_t_ptr);
	gsl_matrix_free(rhs_ptr);
	gsl_matrix_free(xpx);
	gsl_matrix_free(v0_ptr);
	gsl_matrix_free(rhs_t_ptr);
	gsl_matrix_free(tmp1_ptr);
	gsl_matrix_free(tmp2_ptr);
	gsl_matrix_free(tmp4_ptr);
	gsl_matrix_free(mean_matrix_ptr);
	gsl_vector_free(mean_vector_ptr);
	gsl_matrix_free(vare_inv_ptr);
	gsl_matrix_free(varEffects_inv_ptr);
	gsl_matrix_free(lhs_ptr);
	gsl_matrix_free(invLhs_ptr);
	gsl_vector_free(randnum);
	gsl_vector_free(rand_num);
	gsl_vector_free(mean_num);
	gsl_matrix_free(m);
	gsl_matrix_free(rm);
	gsl_rng_free(r);


}//end of the mcmc function


int main(int argc, char *argv[]) {

	int nind, nmarkers, ntraits, numiter, nfold, cycle, thin;
	if(argc<9) {
		printf("usuage: %s filename nind nmarkers ntraits numiter nfold cycle thin\n", argv[0]);
		exit(1);
	} else{
		nind = (int)strtol(argv[2], NULL, 10);
		nmarkers = (int)strtol(argv[3], NULL, 10);
		ntraits = (int)strtol(argv[4], NULL, 10);
		numiter = (int)strtol(argv[5], NULL, 10);
		nfold = (int)strtol(argv[6], NULL, 10);
		cycle = (int)strtol(argv[7], NULL, 10);
		thin= (int)strtol(argv[8], NULL, 10);
	}

	int ntest = (int) nind / nfold, ntrain = nind - ntest;
	int jj, kk, testid[ntest], train_count, test_count;
	gsl_matrix * data = gsl_matrix_calloc(nind, nmarkers + ntraits * 2);
	float num;

	gsl_matrix * x_all = gsl_matrix_calloc(nind, nmarkers + 1);
	gsl_matrix * y_all = gsl_matrix_calloc(nind, ntraits);
	gsl_matrix * a_all = gsl_matrix_calloc(nind, ntraits);
	gsl_matrix * x = gsl_matrix_calloc(ntrain, nmarkers + 1);
	gsl_matrix * y = gsl_matrix_calloc(ntrain, ntraits);
	gsl_matrix * a = gsl_matrix_calloc(ntrain, ntraits);
	gsl_matrix * b = gsl_matrix_calloc(nmarkers + 1, ntraits);
	gsl_matrix * testx = gsl_matrix_calloc(ntest, nmarkers + 1);
	gsl_matrix * testa = gsl_matrix_calloc(ntest, ntraits);
	gsl_matrix * bStore = gsl_matrix_calloc(numiter/thin, (nmarkers + 1) * ntraits);

	FILE *fp;
	char *filename = argv[1];
	fp = fopen(filename, "r");
	for (i = 0; i < nind; i++) {
		for (j = 0; j < nmarkers + ntraits + ntraits; j++) {
			fscanf(fp, "%f", &num);
			gsl_matrix_set(data, i, j, (double) num);
		}
	}
	fclose(fp);

	//initiatize b
	for (i = 0; i < nmarkers + 1; i++) {
		for (j = 0; j < ntraits; j++) {
			gsl_matrix_set(b, i, j, 0);
		}
	}
	//read all genotype into gsl_matrix x_all including the intercept
	for (i = 0; i < nind; i++) {
		gsl_matrix_set(x_all, i, 0, 1);
		for (j = 1; j < nmarkers + 1; j++) {
			gsl_matrix_set(x_all, i, j, gsl_matrix_get(data, i, j-1));
		}
	}
	//read all phenotype and TBV into gsl_matrix y_all and a_all
	for (i = 0; i < nind; i++) {
		for (j = 0; j < ntraits; j++) {
			gsl_matrix_set(y_all, i, j, gsl_matrix_get(data, i, j + nmarkers));
			gsl_matrix_set(a_all, i, j,
					gsl_matrix_get(data, i, j + nmarkers + ntraits));
		}
	}
	//individual id for crossvalidation
	for (i = 0; i < ntest; i++) {
		testid[i] = (cycle - 1) * ntest + i;
	}
	//assign the genotype and phentype for train and test individuals
	train_count = 0;
	test_count = 0;
	for (i = 0; i < nind; i++) {
		if (i >= testid[0] && i <= testid[ntest - 1]) {
			for (jj = 0; jj < nmarkers + 1; jj++) {
				gsl_matrix_set(testx, test_count, jj,
						gsl_matrix_get(x_all, i, jj));
			}
			for (kk = 0; kk < ntraits; kk++) {
				gsl_matrix_set(testa, test_count, kk,
						gsl_matrix_get(a_all, i, kk));
			}
			test_count++;
			continue;
		} else {
			for (j = 0; j < nmarkers + 1; j++) {
				gsl_matrix_set(x, train_count, j, gsl_matrix_get(x_all, i, j));

			}
			for (k = 0; k < ntraits; k++) {
				gsl_matrix_set(y, train_count, k, gsl_matrix_get(y_all, i, k));
				gsl_matrix_set(a, train_count, k, gsl_matrix_get(a_all, i, k));
			}
			train_count++;
		}
	}


	mtbc_mcmc(&thin, &numiter, &ntraits, &ntrain, &nmarkers, y, x, b, bStore);


	gsl_matrix *meanb=gsl_matrix_calloc(nmarkers+1,ntraits);
	gsl_matrix *val_a_matrix=gsl_matrix_calloc(ntest,ntraits);
	double val_a[ntest], test_a[ntest];
	double cor,sum;


	char fileout[100];
	strcpy(fileout, argv[1]);
	strcat(fileout, "_MTBC");
        strcat(fileout, argv[7]);
	fp = fopen(fileout, "w");
	/*
        for (i = 0; i < numiter/thin; i++) {
		for (j = 0; j < ntraits*(nmarkers+1); j++) {
			fprintf(fp, "%lf ", gsl_matrix_get(bStore, i, j));
		}
		fprintf(fp, "\n");
	}
        */
	for(i=0;i<nmarkers+1; i++){
		for(j=0; j<ntraits;j++){
			sum=0;
			for(k=numiter/(2*thin);k<numiter/thin;k++){
				sum+=gsl_matrix_get(bStore,k, i*ntraits+j);
			}
			gsl_matrix_set(meanb, i, j, sum*2*thin/numiter);
		}
	}
	mm_mul(testx,meanb,val_a_matrix);

	for(j=0; j<ntraits; j++){
		for(i=0;i<ntest;i++){
			test_a[i] = gsl_matrix_get(testa,i,j);
			val_a[i] = gsl_matrix_get(val_a_matrix,i,j);
		}
		cor=gsl_stats_correlation(val_a, 1, test_a,1, ntest);
		fprintf(fp, "the accuracy of trait %d prediction is:%lf\n", j+1, cor);
	}

	fclose(fp);


	return 0;
}

