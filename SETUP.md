1. Clone the repository 

git clone <your-repo-url>

cd <your-repo-folder>

2. Set Python Version to 3.11

if on mac and not installed try the following commands

brew install python@3.11

3. Create Virtual Environment

python3.11 -m venv .venv

4. Activate .venv

Mac: 

source .venv/bin/activate

Windows: 

.venv\Scripts\Activate.ps1

5. Install Dependencies 

pip install --upgrade pip

pip install -r requirements.txt

6. Add the Virtual Environment to Jupyter

python -m ipykernel install --user --name=project-env --display-name "Python (project-env)"

7. Launch Jupyter 

jupyter notebook

8. Select the Correct Kernel

Select the kernel: Python (project-env)

9. Run train&test.ipynb as desired

10. To run run.py all you need to do is have python 3.11 and install/activate the venv.