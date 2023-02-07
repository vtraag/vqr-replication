functions {
    #include review_model.stanfunctions
}
data {
    int<lower=0> N_papers; // Number of papers
    int<lower=0> N_institutions; // Number of institutions

    int<lower=0> N_review_scores; // Number of reviews
    int<lower=0> N_citation_scores; // Number of citations scores available

    array[N_papers] int<lower=1,upper=N_institutions> institution_per_paper;

    array[N_review_scores] int<lower=1,upper=28> review_score; // Review score per paper
    array[N_review_scores] int<lower=1,upper=N_papers> paper_per_review_score;

    array[N_citation_scores] real<lower=0> citation_score; // Citation score per paper
    array[N_citation_scores] int<lower=1,upper=N_papers> paper_per_citation_score;

    // Toggle to determine whether citation_scores represent percentile scores
    // or whether they represent (normalised) citation scores.
    int citation_percentile_score;

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

    real general_paper_value_sigma = 1.3;
    real general_paper_value_mu = -general_paper_value_sigma^2/2;

    // Cutpoints for the distribution of the review scores
    int K_review_score_points = 28;
    ordered[K_review_score_points-1] review_cutpoints;

    for (i in 1:(K_review_score_points - 1))
    {
        review_cutpoints[i] = exp(general_paper_value_mu + general_paper_value_sigma*inv_Phi( to_real(i)/K_review_score_points ));
    }

    array[N_citation_scores] real<lower=0> raw_citation_score;

    if (citation_percentile_score)
    {
        // Assume that the percentile comes from the same distribution
        // as the review score distribution, and use that to transform
        // the percentil score to a "raw" score (except for the fact that
        // we expect it to be normalised).
        for (i in 1:N_citation_scores)
        {
            real percentile = citation_score[i] / 100.0;
            
            // We threshold the percentile to avoid infinite values
            if (percentile <= 0)
            {
                percentile = 0.001;
            }            
            if (percentile >= 1)
            {
                percentile = 0.999;
            }

            raw_citation_score[i] = exp(general_paper_value_mu + general_paper_value_sigma*inv_Phi( percentile ));
        }
    }
    else
        raw_citation_score = citation_score;
}
parameters {
    // Review value per paper
    vector[N_papers] value_per_paper_raw;

    // Citation value for each institute
    vector[N_institutions] value_inst;
    // Citation value for each institute
    vector<lower=0>[N_institutions] sigma_value_inst;

    // Coefficient of citation
    real beta;

    // Standard deviation of citation
    real<lower=0> sigma_cit;

    // Standard deviation of peer review.
    real<lower=0> sigma_review;

    real beta_nonzero_cit;
}
transformed parameters {
    # Use non-centered parameterization
    vector<lower=0>[N_papers] value_per_paper;
    value_per_paper = exp(value_per_paper_raw .* sigma_value_inst[institution_per_paper] + value_inst[institution_per_paper]);
}
model {

    if (use_estimated_priors)
    {
        sigma_review ~ normal(sigma_review_mu, sigma_review_sigma);
        sigma_cit ~ normal(sigma_cit_mu, sigma_cit_sigma);

        beta ~ normal(beta_mu, beta_sigma);

        beta_nonzero_cit ~ normal(beta_nonzero_cit_mu, beta_nonzero_cit_sigma);
    }
    else
    {
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
        sigma_value_inst ~ exponential(1);

        // The review and citation value for each paper is sampled from a normal
        // distribution centered at the review and citations values for the
        // institutions that the papers is a part of, with a certain correlation
        // between the review and the citation value.
        value_per_paper_raw ~ std_normal();
    }

    for (i in 1:N_citation_scores)
    {
        real value = value_per_paper[paper_per_citation_score[i]];

        raw_citation_score[i] ~ hurdle_lognormal_logit(beta*log(value) - sigma_cit^2/2, 
                                                   sigma_cit,
                                                   beta_nonzero_cit*log(value));
    }

    // The actual review scores per paper are sampled from a normal distribution
    // which is centered at the citation value for each paper, with a certain
    // uncertainty.
    for (i in 1:N_review_scores)
    {
        real value = value_per_paper[paper_per_review_score[i]];

        review_score[i] ~ ordinal_lognormal(log(value) - sigma_review^2/2,
                                            sigma_review,
                                            review_cutpoints);
    }
}
generated quantities {
    array[N_papers] int review_score_ppc;
    array[N_papers] real citation_ppc;

    for (i in 1:N_papers)
    {
        real value = value_per_paper[i];

        review_score_ppc[i] = ordinal_lognormal_rng(log(value) - sigma_review^2/2, 
                                                    sigma_review, 
                                                    review_cutpoints);

        citation_ppc[i] = hurdle_lognormal_logit_rng(beta*log(value) - sigma_cit^2/2,
                                                     sigma_cit,
                                                     beta_nonzero_cit*log(value));

        if (citation_percentile_score)
        {
            citation_ppc[i] = lognormal_cdf(citation_ppc[i], general_paper_value_mu, general_paper_value_sigma);
        }
    }
}