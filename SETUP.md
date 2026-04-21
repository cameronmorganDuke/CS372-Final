1. Clone the repository 

git clone <your-repo-url>

cd <your-repo-folder>

2. If .venv is not already included or if you want your own

python3 -m venv .venv

3. Activate .venv

Mac: 

source .venv/bin/activate

Windows: 

.venv\Scripts\Activate.ps1

4. Install Dependencies 

pip install --upgrade pip

pip install -r requirements.txt

5. Add the Virtual Environment to Jupyter

python -m ipykernel install --user --name=project-env --display-name "Python (project-env)"

6. Launch Jupyter 

jupyter notebook

7. Select the Correct Kernel

Select the kernel: Python (project-env)