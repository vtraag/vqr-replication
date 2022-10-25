data {
    int<lower=0> N_reviews; // Number of reviews
    int<lower=0> N_papers; // Number of papers
    int<lower=0> N_institutions; // Number of institutions

    array[N_reviews] real<lower=3,upper=30> review_score; // Review score per paper
    array[N_reviews] int<lower=1,upper=N_papers> paper_per_review;

    array[N_papers] real<lower=0> citation_score; // Citation score
    array[N_papers] int<lower=1,upper=N_institutions> institution_per_paper;
}
parameters {
    // Review value per paper
    vector[N_papers] value_paper_rev;

    // Citation value for each institute
    vector[N_institutions] value_inst_cit;

    // Review value for each institute
    vector[N_institutions] value_inst_rev;

    // Correlation between review and citation value at paper level
    corr_matrix[2] corr_paper;

    // Correlation between review and citation value at institutional level
    corr_matrix[2] corr_inst;

    // Scale (i.e. standard deviation) for review and citation at paper level
    vector<lower=0>[2] scale_paper;

    // Scale (i.e. standard deviation) for review and citation at institutional level
    vector<lower=0>[2] scale_inst;

    // Standard deviation of peer review.
    real<lower=0> sigma_review;
}
model {

    // Priors for scales and correlations
    corr_paper ~ lkj_corr(2);
    corr_inst  ~ lkj_corr(2);

    scale_paper ~ cauchy(0, 2.5);
    scale_inst  ~ cauchy(0, 2.5);

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
        value_inst ~ multi_normal(to_vector([0,0]),
                                  quad_form_diag(corr_inst, scale_inst));

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
        cit_value_rev ~ multi_normal(value_inst[institution_per_paper,],
                                     quad_form_diag(corr_paper, scale_paper));
    }

    // The actual review scores per paper are sampled from a normal distribution
    // which is centered at the citation value for each paper, with a certain
    // uncertainty.
    review_score ~ normal(value_paper_rev[paper_per_review], sigma_review);

}