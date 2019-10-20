#include <jni.h>
#include <string>
#include "parameters.h"

#include <stdio.h>
// #include <malloc.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "parameters.h"

extern "C"{
void REF_MMult(int, int, int, int8_t*, int, int8_t *, int, int32_t *, int );
void MY_MMult(int, int, int, int8_t *, int, int8_t *, int, int32_t *, int, double*, double*, double*);
void copy_int8_matrix(int, int, int8_t *, int, int8_t *, int );
void copy_int32_matrix(int, int, int32_t *, int, int32_t *, int );
void random_int8_matrix(int, int, int8_t *, int);
int32_t compare_matrices( int, int, int32_t *, int, int32_t *, int );
double dclock();
}

int test_MMult()
{
    int     ret = 0;
    int
            p,
            m, n, k,
            lda, ldb, ldc,
            rep;

    double
            dtime, dtime_best,
            gflops;

    double
            packZ, packN, kernel;

    int32_t
            diff;

    int8_t
            *a, *b;

    int32_t
            *c, *cref, *cold;

    printf( "MY_MMult = [\n" );

    for ( p=PFIRST; p<=PLAST; p+=PINC ){
        m = n = k = p;
        gflops = 2.0 * m * n * k * 1.0e-09;

        lda = k;
        ldb = n;
        ldc = n;

        /* Allocate space for the matrices */
        /* Note: I create an extra column in A to make sure that
           prefetching beyond the matrix does not cause a segfault */
        a = ( int8_t * ) malloc( m * k * sizeof( int8_t ) );
        b = ( int8_t * ) malloc( k * n * sizeof( int8_t ) );
        c = ( int32_t * ) malloc( m * n * sizeof( int32_t ) );
        cold = ( int32_t * ) malloc( m * n * sizeof( int32_t ) );
        cref = ( int32_t * ) malloc( m * n * sizeof( int32_t ) );

        /* Generate random matrices A, B, Cold */
        random_int8_matrix( m, k, a, lda );
        random_int8_matrix( k, n, b, ldb );
#if 1
        memset(cold, 0, m * n * sizeof(int32_t));
#endif

        copy_int32_matrix( m, n, cold, ldc, cref, ldc );

        /* Run the reference implementation so the answers can be compared */

        REF_MMult( m, n, k, a, lda, b, ldb, cref, ldc );

        /* Time the "optimized" implementation */
        packZ = packN = kernel = 0.0;
        for ( rep=0; rep<NREPEATS; rep++ ){
            copy_int32_matrix( m, n, cold, ldc, c, ldc );

            /* Time your implementation */
            dtime = dclock();

            MY_MMult( m, n, k, a, lda, b, ldb, c, ldc, &packZ, &packN, &kernel);

            dtime = dclock() - dtime;

            if ( rep==0 )
                dtime_best = dtime;
            else
                dtime_best = ( dtime < dtime_best ? dtime : dtime_best );
        }

        diff = compare_matrices( m, n, c, ldc, cref, ldc );
        if(diff != 0){
            ret = -1;
        }

        printf( "%d %0.3lf %d \n", p, gflops / dtime_best, diff );
//    double sum = packZ + packN + kernel;
//    printf( "packZ: %0.2f, packN: %0.2lf, kernel: %0.2lf, sum: %0.2lf \n", packZ, packN, kernel, sum);
        fflush( stdout );

        free( a );
        free( b );
        free( c );
        free( cold );
        free( cref );
    }

    printf( "];\n" );
    return ret;
//  exit( 0 );
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_myapplication_MainActivity_stringFromJNI(
        JNIEnv* env,
        jobject /* this */) {
    std::string hello = "GEMM test passed";

    int iRet = test_MMult();
    if(iRet != 0) {
        hello = "result not match";
    }

    return env->NewStringUTF(hello.c_str());
}





