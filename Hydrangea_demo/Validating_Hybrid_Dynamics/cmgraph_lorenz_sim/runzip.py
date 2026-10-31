pip install -r requirements.txt
export PYTHONPATH=$PWD
python run_hopf.py 256 out 512    # Hopf: 4 vs 10 steps at 256², plus 10 steps at 512²
python run_lorenz.py 40 out       # Lorenz 3D at 40³
python run_scaling.py out         # timing sweep
python make_report.py out report.html
pytest -q tests