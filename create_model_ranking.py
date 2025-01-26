import model_tester as test

if __name__ == "__main__":

    # Ranks all models in models/initial_model_20epochs.
    directory = "models/initial_model_20epochs"

    # Lists directories to avoid.
    excepted_dirs = ["5seq-19epochs-mse-32batch-351neurons-0_01lr-351features-0dr",
                     "5seq-20epochs-mse-32batch-351neurons-0_01lr-351features-0dr"]
    
    # Runs function to rank models by r2 score for test set.
    test.rank_rootdir_models(directory=directory, dataset="test", excepted_dirs=excepted_dirs)

    # Runs function to rank models by r2 score for training set.
    test.rank_rootdir_models(directory=directory, dataset="training", excepted_dirs=excepted_dirs)