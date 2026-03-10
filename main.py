import subprocess
import sys
import time



def main():
    start = time.time()
    scripts = [
        "src/data_generation.py",
        "src/train_model.py",
        "src/user_profiles.py",
        "src/evaluate_model.py"
    ]

    for script in scripts:
        print(f"=== Running {script} ===")
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            print(f"Error executing {script}. Exiting pipeline.")
            sys.exit(result.returncode)
            
    print("Pipeline executed successfully.")

    print(f"Pipeline finished in {time.time() - start:.2f} seconds")

if __name__ == "__main__":
    main()