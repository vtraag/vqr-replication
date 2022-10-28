functions {
    real to_real(int x)
    {
        return 1.0*x;
    }

    real review_score_logpdf(int review_score, vector cutpoints, real mu, real sigma)
    {
        if (review_score <= 1)
        {
            // This would be similar to having review_score[0] = -Infinity
            return normal_lcdf(cutpoints[1] | mu, sigma);
        }
        else if (review_score > size(cutpoints))
        {
            // This would be similar to having review_score[size(cutpoints) + 1] = Infinity
            return normal_lccdf(cutpoints[size(cutpoints)] | mu, sigma);
        }
        else
        {
            return log_diff_exp(normal_lcdf(cutpoints[review_score] | mu, sigma),
                                normal_lcdf(cutpoints[review_score - 1]     | mu, sigma));
        }
    }

}
data {
    int<lower=0> N_reviews; // Number of reviews
    int<lower=0> N_papers; // Number of papers
    int<lower=0> N_institutions; // Number of institutions

    array[N_reviews] int<lower=1,upper=28> review_score; // Review score per paper
    array[N_reviews] int<lower=1,upper=N_papers> paper_per_review;

    array[N_papers] real<lower=0> citation_score; // Citation score
    array[N_papers] int<lower=1,upper=N_institutions> institution_per_paper;
}
transformed data {
    // Cutpoints for the distribution of the review scores
    int K_review_score_points = 28;
    ordered[K_review_score_points-1] review_cutpoints;
    for (i in 1:(K_review_score_points - 1))
    {
        review_cutpoints[i] = inv_Phi( to_real(i)/K_review_score_points );
    }
}
parameters {
    // Review value per paper
    vector[N_papers] value_paper_rev;

    // Citation value for each institute
    vector[N_institutions] value_inst_cit;

    // Review value for each institute
    vector[N_institutions] value_inst_rev;

    // Cholesky factor for correlation between review and citation value at paper level
    cholesky_factor_corr[2] L_corr_paper;

    // Cholesky factor for correlation between review and citation value at institutional level
    cholesky_factor_corr[2] L_corr_inst;

    // Scale (i.e. standard deviation) for review and citation at paper level
    vector<lower=0>[2] scale_paper;

    // Scale (i.e. standard deviation) for review and citation at institutional level
    vector<lower=0>[2] scale_inst;

    // Standard deviation of peer review.
    real<lower=0> sigma_review;
}
transformed parameters {

    // Correlation between review and citation value at paper level
    corr_matrix[2] corr_paper = multiply_lower_tri_self_transpose(L_corr_paper);

    // Correlation between review and citation value at institutional level
    corr_matrix[2] corr_inst = multiply_lower_tri_self_transpose(L_corr_inst);
}
model {

    // Priors for scales and correlations
    L_corr_paper ~ lkj_corr_cholesky(2);
    L_corr_inst  ~ lkj_corr_cholesky(2);

    scale_paper ~ exponential(1);
    scale_inst  ~ exponential(1);

    sigma_review ~ exponential(1);

    {
        // Build up vectorised representation for multidimensional normal distribution
        array[N_institutions] vector[2] value_inst;
        for (i in 1:N_institutions)
        {
            value_inst[i] = to_vector([value_inst_rev[i], value_inst_cit[i]]);
        }

        // The review and citation value for each institution is sampled from a
        // normal distribution centered at 0, with a certain correlation between
        // the review and the citation value.
        value_inst ~ multi_normal_cholesky(to_vector([0,0]), diag_pre_multiply(scale_inst, L_corr_inst));

        // Build up vectorised representation for multidimensional normal distribution
        array[N_papers] vector[2] cit_value_rev;
        for (i in 1:N_papers)
        {
            cit_value_rev[i] = to_vector([value_paper_rev[i], citation_score[i]]);
        }

        // The review and citation value for each paper is sampled from a normal
        // distribution centered at the review and citations values for the
        // institutions that the papers is a part of, with a certain correlation
        // between the review and the citation value.
        cit_value_rev ~ multi_normal_cholesky(value_inst[institution_per_paper,],
                                              diag_pre_multiply(scale_paper, L_corr_paper));
    }

    // The actual review scores per paper are sampled from a normal distribution
    // which is centered at the citation value for each paper, with a certain
    // uncertainty.
    for (i in 1:N_reviews)
    {
        real rev_score_p = review_score_logpdf(review_score[i], 
                                      review_cutpoints,
                                      value_paper_rev[paper_per_review[i]],
                                      sigma_review);
        target += rev_score_p;
    }
}