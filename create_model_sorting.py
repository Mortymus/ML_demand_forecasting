import model_tester as test

if __name__ == "__main__":

    # Ranks all models in models/initial_model_20epochs.
    directory = "models"

    # Lists directories to avoid.
    excepted_dirs = ["testing", "revised_article_removal", "revised_article_testing"]
    
    # Runs function to sort models by hyperparameters.
    test.sort_rootdir_models(directory=directory, excepted_dirs=excepted_dirs)