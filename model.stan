functions {
    // From Statistical Rethinking
    vector merge_missing(array[] int miss_indexes, vector x_obs, vector x_miss)
    {
      int N = dims(x_obs)[1];
      int N_miss = dims(x_miss)[1];
      vector[N] merged;
      merged = x_obs;
      for (i in 1:N_miss)
      {
          merged[ miss_indexes[i] ] = x_miss[i];
      }
      return merged;
    }
}
data {
    int<lower=0> N; // Number of papers
    int<lower=0> N_institutions; // Number of institutions

    /* Reviewer 1 */
    array[N] real<lower=3,upper=30> review_score_1; // Review score per paper
    int N_review_score_1_missing; // Number of review scores missing
    array[N_review_score_1_missing] int<lower=1,upper=N> review_score_1_missing_idx; // Missing review scores indices

    /* Reviewer 2 */
    array[N] real<lower=3,upper=30> review_score_2; // Review score per paper
    int N_review_score_2_missing; // Number of review scores missing
    array[N_review_score_2_missing] int<lower=1,upper=N> review_score_2_missing_idx; // Missing review scores indices

    array[N] real<lower=0> citation_score; // Citation score
    int N_citation_score_missing; // Number of citation scores missing
    array[N_citation_score_missing] int<lower=1,upper=N>  citation_score_missing_idx; // Missing citation score indices
    int citation_score_type; // Citation score type (i.e. NCS/NJS or Percentile Score)

    array[N] int<lower=1,upper=N_institutions> institution_per_paper;
}
transformed data {
   array[N] vector[3] metrics;
   metrics[1] = to_vector(citation_score);
   metrics[2] = to_vector(review_score_1);
   metrics[3] = to_vector(review_score_2);
}
parameters {
    real<lower=0> sigma_paper_cit;
    real<lower=0> sigma_paper_review;
    real<lower=0> corr_paper_review_cit;

    vector[N] value_paper_cit;
    vector[N] value_paper_rev;

    real<lower=0> sigma_inst_cit;
    real<lower=0> sigma_inst_review;
    real<lower=0> corr_inst_review_cit;

    vector[N_institutions] value_inst_cit;
    vector[N_institutions] value_inst_rev;
}
transformed parameters {
    matrix[3,3] sigma_paper = [[sigma_paper_cit, corr_paper_review_cit, corr_paper_review_cit],
                               [corr_paper_review_cit, sigma_paper_review, sigma_paper_review],
                               [corr_paper_review_cit, sigma_paper_review, sigma_paper_review]];

    matrix[3,3] sigma_inst = [[sigma_inst_cit, corr_inst_review_cit, corr_inst_review_cit],
                              [corr_inst_review_cit, sigma_inst_review, sigma_inst_review],
                              [corr_inst_review_cit, sigma_inst_review, sigma_inst_review]];

    array[N] vector[2] values;
    values[1] = value_paper_cit;
    values[2] = value_paper_rev;
}
model {
    // For NCS-like citation scores
    citation_score, review_score_1, review_score_2 ~ multilognormal([value_paper_cit, value_paper_rev, value_paper_rev], 
                                                                    sigma_paper);

    value_paper_cit, value_paper_rev ~ multilognormal([value_inst_cit[institution_per_paper], value_inst_rev[institution_per_paper]],
                                                      sigma_inst);


}