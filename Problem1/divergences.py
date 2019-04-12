import numpy as np
import scipy.stats
import scipy as sp


def jensen_shannon_divergence(p, q):
    p, q = np.asarray(p), np.asarray(q)
    p, q = p/p.sum(), q/q.sum()
    m = 0.5*(p + q)
    return sp.stats.entropy(p, m, np.e)/2.0 + sp.stats.entropy(q, m, np.e)/2.0


def wasserstein_distance():
    """
    int N= 10;
    int threshold= 3;

    double[] P= randomDoubleArray(N);
    double[] Q= randomDoubleArray(N);

    double[][] C= new double[N][N];
    for (int i=0; i<N; ++i) {
        for (int j=0; j<N; ++j) {
            int abs_diff= Math.abs(i-j);
            C[i][j]= Math.min(abs_diff,threshold);
        }
    }

    double extra_mass_penalty= -1;

    double dist= emd_hat.dist_gd_metric(P,Q,C,extra_mass_penalty,null);

    System.out.print("Distance==");
    System.out.println(dist);


    public static double dist_gd_metric(
        double[] P, double[] Q, double[][] C, double extra_mass_penalty,
        double[][] F) {
        double dist= dist_compute(P,Q,C,extra_mass_penalty, F, true);
        return dist;
    } // dist_gd_metric

    private static double dist_compute(
        double[] P, double[] Q, double[][] C, double extra_mass_penalty,
        double[][] F,
        boolean gd_metric) {

        int N= P.length;

        double[] Cv= convert_to_row_by_row(C);
        double[] Fv= null;
        if (F!=null) Fv= convert_to_row_by_row(F);

        int Fv_is_null_int= 0;
        if (Fv==null) Fv_is_null_int= 1;

        int gd_metric_int= 0;
        if (gd_metric) gd_metric_int= 1;

        double dist= native_dist_compute(P,Q,Cv,extra_mass_penalty, Fv, N,gd_metric_int,Fv_is_null_int);
        if (F!=null) convert_back_from_row_by_row(Fv,F);
        return dist;
    } // dist_compute

    private static native double native_dist_compute(
        double[] P, double[] Q, double[] Cv, double extra_mass_penalty,
        double[] Fv,
        int N, int gd_metric, int Fv_is_null);


    """
    pass


if __name__ == '__main__':
    # Test
    print(jensen_shannon_divergence([0.6, 0.4], [0.5, 0.5]))
