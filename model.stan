functions {
    real to_real(int x)
    {
        return 1.0*x;
    }

    real ordinal_normal_lpmf(int review_score, real mu, real sigma, vector cutpoints)
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

    int ordinal_normal_rng(real mu, real sigma, vector cutpoints)
    {
        int K = size(cutpoints) + 1;
        vector[K] prob;
        vector[K - 1] cum_prob;
        
        // Get cumulative probabilities
        for (i in 1:K - 1)
            cum_prob[i] = normal_cdf(cutpoints[i] | mu, sigma);

        // Get probabilities between cutpoints
        prob[1] = cum_prob[1];
        prob[2:K - 1] = cum_prob[2:K - 1] - cum_prob[1:K - 2];
        prob[K] = 1 - cum_prob[K - 1];

        // Sample random element with individual probability
        return categorical_rng(prob);
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
    vector[N_papers] value_paper;

    // Citation value for each institute
    vector[N_institutions] value_inst;

    real<lower=0> sigma_paper_value;

    // Coefficient of citation
    real<lower=0> beta;

    // Standard deviation of citation
    real<lower=0> sigma_cit;

    // Standard deviation of peer review.
    real<lower=0> sigma_review;
}
model {

    sigma_paper_value ~ exponential(1);
    sigma_review ~ exponential(1);
    sigma_cit ~ exponential(1);

    beta ~ exponential(1);

    {
        // The review and citation value for each institution is sampled from a
        // normal distribution centered at 0, with a certain correlation between
        // the review and the citation value.
        value_inst ~ normal(0, 1);

        // The review and citation value for each paper is sampled from a normal
        // distribution centered at the review and citations values for the
        // institutions that the papers is a part of, with a certain correlation
        // between the review and the citation value.
        value_paper ~ normal(value_inst[institution_per_paper], sigma_paper_value);
    }

    citation_score ~ normal(beta*value_paper, sigma_cit);

    // The actual review scores per paper are sampled from a normal distribution
    // which is centered at the citation value for each paper, with a certain
    // uncertainty.
    for (i in 1:N_reviews)
    {
        review_score[i] ~ ordinal_normal(value_paper[paper_per_review[i]],
                                         sigma_review,
                                         review_cutpoints);
    }
}
generated quantities {
    array[N_papers] int review_score_ppc;

    for (i in 1:N_papers)
    {
        review_score_ppc[i] = ordinal_normal_rng(value_paper[i], sigma_review, review_cutpoints);
    }
}