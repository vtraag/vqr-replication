functions {
    #include review_model.stanfunctions
}
data {
    int<lower=0> N_reviews; // Number of reviews
    int<lower=0> N_papers; // Number of papers
    int<lower=0> N_institutions; // Number of institutions

    array[N_reviews] int<lower=1,upper=28> review_score; // Review score per paper
    array[N_reviews] int<lower=1,upper=N_papers> paper_per_review;

    array[N_papers] real<lower=0> citation_score; // Citation score
    array[N_papers] int<lower=1,upper=N_institutions> institution_per_paper;

    // Toggle to determine whether we want to estimate coefficients
    // using the overall priors, or whether we want to use priors that
    // are based on estimates themselves.
    // 
    // If we use already estimated coefficients, we can use this
    // to predict scores through paper values. If we would not use 
    // estimated coefficients, this would lead to inferior prediction,
    // and the parameters could not be leanred without acccess to both scores.
    int use_estimated_priors;

    // The below are estimated coefficients from other models.
    real<lower=0> sigma_paper_value_mu;
    real<lower=0> sigma_paper_value_sigma;

    // Coefficient of citation
    real beta_mu;
    real<lower=0> beta_sigma;

    // Standard deviation of citation
    real<lower=0> sigma_cit_mu;
    real<lower=0> sigma_cit_sigma;

    // Standard deviation of peer review.
    real<lower=0> sigma_review_mu;
    real<lower=0> sigma_review_sigma;

    real beta_nonzero_cit_mu;
    real<lower=0> beta_nonzero_cit_sigma;

}
transformed data {
    // Cutpoints for the distribution of the review scores
    int K_review_score_points = 28;
    ordered[K_review_score_points-1] review_cutpoints;
    for (i in 1:(K_review_score_points - 1))
    {
        review_cutpoints[i] = exp(inv_Phi( to_real(i)/K_review_score_points ));
    }
}
parameters {
    // Review value per paper
    vector<lower=0>[N_papers] value_paper;

    // Citation value for each institute
    vector[N_institutions] value_inst;

    real<lower=0> sigma_paper_value;

    // Coefficient of citation
    real beta;

    // Standard deviation of citation
    real<lower=0> sigma_cit;

    // Standard deviation of peer review.
    real<lower=0> sigma_review;

    real beta_nonzero_cit;    
}
model {

    if (use_estimated_priors)
    {
        sigma_paper_value ~ normal(sigma_paper_value_mu, sigma_paper_value_sigma);
        sigma_review ~ normal(sigma_review_mu, sigma_review_sigma);
        sigma_cit ~ normal(sigma_cit_mu, sigma_cit_sigma);

        beta ~ normal(beta_mu, beta_sigma);

        beta_nonzero_cit ~ normal(beta_nonzero_cit_mu, beta_nonzero_cit_sigma);
    }
    else
    {
        sigma_paper_value ~ exponential(1);
        sigma_review ~ exponential(1);
        sigma_cit ~ exponential(1);

        beta ~ normal(0, 1);

        beta_nonzero_cit ~ normal(0, 1);    
    }

    {
        // The review and citation value for each institution is sampled from a
        // normal distribution centered at 0, with a certain correlation between
        // the review and the citation value.
        value_inst ~ normal(0, 1);

        // The review and citation value for each paper is sampled from a normal
        // distribution centered at the review and citations values for the
        // institutions that the papers is a part of, with a certain correlation
        // between the review and the citation value.
        value_paper ~ lognormal(value_inst[institution_per_paper], sigma_paper_value);
    }

    for (i in 1:N_papers)
    {
        citation_score[i] ~ hurdle_lognormal_logit(beta*log(value_paper[i]) - sigma_cit^2/2, sigma_cit, beta_nonzero_cit*value_paper[i]);
    }

    // The actual review scores per paper are sampled from a normal distribution
    // which is centered at the citation value for each paper, with a certain
    // uncertainty.
    for (i in 1:N_reviews)
    {
        review_score[i] ~ ordinal_lognormal(log(value_paper[paper_per_review[i]]) - sigma_review^2/2,
                                         sigma_review,
                                         review_cutpoints);
    }
}
generated quantities {
    array[N_papers] int review_score_ppc;
    array[N_papers] real citation_ppc;

    for (i in 1:N_papers)
    {
        review_score_ppc[i] = ordinal_lognormal_rng(log(value_paper[i]) - sigma_review^2/2, sigma_review, review_cutpoints);


        citation_ppc[i] = hurdle_lognormal_logit_rng(beta*log(value_paper[i]) - sigma_cit^2/2,
                                            sigma_cit,
                                            beta_nonzero_cit*value_paper[i]);

    }
}