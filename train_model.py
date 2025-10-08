import importlib
import os
import sys

def main():
    print("🏎️ Formula 1 Race Predictor 🏎️")
    print("--------------------------------------")

    # Ask race name from user
    race_name = input("Enter race name (e.g. japan, australia, bahrain): ").strip().lower()

    # Build module and file paths
    module_path = f"Predictions.{race_name}_gp"
    expected_file = os.path.join("Predictions", f"{race_name}_gp.py")
    cache_path = os.path.join("Predictions", f"cache_{race_name}")

    # Check if race file exists
    if not os.path.exists(expected_file):
        print(f"❌ Error: '{expected_file}' not found.")
        print("Make sure your file is named like 'japan_gp.py' and inside the 'Predictions' folder.")
        return

    # Ensure cache directory exists
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
        print(f"📁 Created missing cache directory: {cache_path}")

    try:
        print(f"\n⚙️ Running prediction for {race_name.capitalize()} Grand Prix...\n")
        race_module = importlib.import_module(module_path)

        # Handle race prediction execution
        if hasattr(race_module, "run_prediction"):
            try:
                # First try calling with cache path
                result = race_module.run_prediction(cache_path)
            except TypeError:
                # If it doesn't need a cache path, call without arguments
                result = race_module.run_prediction()
            
            print("\n✅ Prediction completed successfully!\n")

            # Show result if available
            if result is not None:
                print("🏁 Prediction Result:")
                print(result)
            else:
                print("ℹ️ No explicit result returned — check your race script’s print statements.")

        else:
            print(f"✅ {race_name}_gp.py loaded successfully.")
            print("ℹ️ No 'run_prediction()' function found — executed full script directly.")

    except Exception as e:
        print(f"🚨 Error running prediction for {race_name.capitalize()} GP: {e}")

if __name__ == "__main__":
    main()  