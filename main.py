# main.py

from scripts import wrangle, model_training, validate, evaluate

def run_pipeline():
    print("\nStep 1: Data Wrangling")
    wrangle.main()

    print("\nStep 2: Model Training")
    model_training.main()

    print("\nStep 3: Walk-Forward Validation")
    validate.main()

    print("\nStep 4: Evaluation & Visualization")
    evaluate.main()

    print("\nPipeline completed successfully.")

if __name__ == "__main__":
    run_pipeline()
